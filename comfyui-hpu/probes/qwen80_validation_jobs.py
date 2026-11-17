"""Validate readback buffering, then restore the original ordinary service."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

import psutil
import qwen21_latest_jobs as lifecycle

ROOT = lifecycle.BASE / 'performance-qwen80-2026-11-16/product-validation'
lifecycle.ROOT = ROOT
FILES = ['comfy/text_encoders/llama.py', 'comfy/text_encoders/qwen35.py',
         'tests-unit/comfy_test/generation_readback_test.py',
         'tests-unit/comfy_test/hpu_qwen_generation_test.py']
URL = 'http://127.0.0.1:8190'


def request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(URL + path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        content = response.read()
    return json.loads(content) if content else None


def workflows(environment):
    directory = ROOT / 'api'
    directory.mkdir(exist_ok=True)
    log = (directory / 'server.log').open('w')
    server = subprocess.Popen([sys.executable, '-c', 'import qwen80_server; qwen80_server.main()',
        '--disable-dynamic-vram', '--disable-pinned-memory', '--disable-async-offload',
        '--disable-xformers', '--use-pytorch-cross-attention', '--disable-all-custom-nodes',
        '--listen', '127.0.0.1', '--port', '8190'], cwd=lifecycle.PRODUCT,
        env=dict(environment, MOSS_EVENTS=str(directory / 'events.jsonl')), stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            if server.poll() is not None:
                raise RuntimeError('Validation server exited during startup')
            try:
                request('/queue')
                break
            except urllib.error.URLError:
                time.sleep(1)
        else:
            raise TimeoutError('Validation server startup')
        sequence = [('cold', 4)]
        for i, size in enumerate([1, 4, 4, 1, 1, 4, 4, 1, 1, 4]):
            sequence.append((f'warm-{i}-readback{size}', size))
        sequence += [('small', 4), ('reload', 4)]
        original = json.loads(lifecycle.WORKFLOW.read_text())
        for label, size in sequence:
            if label == 'reload':
                request('/free', dict(unload_models=True, free_memory=True))
                samples = []
                for _ in range(5):
                    time.sleep(10)
                    samples.append(request('/system_stats'))
                (directory / 'unload.json').write_text(json.dumps(samples, indent=2))
            graph = copy.deepcopy(original)
            if label == 'small':
                graph['13']['inputs'].update(aspect_ratio='1:1 (Square)', megapixels=.25)
                graph['459:471']['inputs']['sampling_mode.seed'] += 1
                graph['459:488']['inputs']['noise_seed'] += 1
            graph['461']['inputs']['filename_prefix'] = 'gaudi/qwen80/' + label
            pid = request('/prompt', dict(prompt=graph, extra_data=dict(qwen_readback=size,
                moss_recompute_nodes=['TextGenerate', 'TextEncodeQwenImage21', 'SamplerCustomAdvanced', 'VAEDecode'])))['prompt_id']
            print(label, pid, flush=True)
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError('Validation server exited during execution')
                history = request('/history/' + pid)
                if pid in history:
                    break
                time.sleep(.5)
            else:
                raise TimeoutError(pid)
            result = history[pid]
            (directory / (label + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
            assert result['status']['status_str'] == 'success', result['status']
            stamps = [m[1]['timestamp'] for m in result['status']['messages'] if m[0] in ('execution_start', 'execution_success')]
            print(label, 'seconds', (stamps[-1] - stamps[0]) / 1000, flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=40)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        (directory / 'server-exit.json').write_text(json.dumps({'returncode': server.returncode}))
        log.close()


def main():
    ROOT.mkdir(exist_ok=True)
    deadline = time.monotonic() + 1800
    while psutil.pid_exists(151076):
        if time.monotonic() > deadline:
            raise TimeoutError('Readback screening still owns the card')
        time.sleep(2)
    queue = lifecycle.request('/queue')
    assert not queue['queue_running'] and not queue['queue_pending'], queue
    environment = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
                       PYTHONPATH=os.pathsep.join((str(lifecycle.PRODUCT), str(lifecycle.PROBES))))
    backup = ROOT / 'baseline-source'
    backup.mkdir(exist_ok=True)
    present = []
    for name in FILES:
        path = lifecycle.PRODUCT / name
        if path.exists():
            target = backup / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            present.append(name)
    try:
        subprocess.run(['./stop-hpu.sh'], cwd=lifecycle.PRODUCT, check=True)
        for name in FILES:
            shutil.copy2(ROOT / 'candidate-source' / name, lifecycle.PRODUCT / name)
        tests = [FILES[2], FILES[3], 'tests-unit/comfy_test/qwen35_decode_test.py',
                 'tests-unit/comfy_test/hpu_qwen_norm_test.py', 'tests-unit/comfy_test/hpu_qwen21_block_test.py']
        lifecycle.run('tests', [sys.executable, '-m', 'pytest', '-q'] + tests, environment)
        workflows(environment)
    finally:
        for name in FILES:
            path = lifecycle.PRODUCT / name
            if name in present:
                shutil.copy2(backup / name, path)
            else:
                path.unlink(missing_ok=True)
        subprocess.run(['./start-hpu.sh'], cwd=lifecycle.PRODUCT, check=True, timeout=180)
        lifecycle.verify_restoration()


if __name__ == '__main__':
    main()
