"""Validate a product snapshot and interleave fixed-workload API comparisons."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error

import numpy as np
from PIL import Image
import psutil
import qwen21_latest_jobs as lifecycle
from qwen80_validation_jobs import request


def workflows(root, environment):
    api = root / 'api'
    api.mkdir()
    environment = dict(environment, QWEN80_VALIDATION=str(root), MOSS_EVENTS=str(api / 'events.jsonl'))
    log = (api / 'server.log').open('w')
    server = subprocess.Popen([sys.executable, '-c', 'import qwen80_pe_server; qwen80_pe_server.main()',
        '--disable-dynamic-vram', '--disable-pinned-memory', '--disable-async-offload',
        '--disable-xformers', '--use-pytorch-cross-attention', '--disable-all-custom-nodes',
        '--listen', '127.0.0.1', '--port', '8190'], cwd=lifecycle.PRODUCT,
        env=environment, stdout=log, stderr=subprocess.STDOUT)
    results = []
    try:
        for _ in range(120):
            assert server.poll() is None, 'Validation server exited'
            try:
                request('/queue')
                break
            except urllib.error.URLError:
                time.sleep(1)
        else:
            raise TimeoutError('Validation startup')
        graph = json.loads(lifecycle.WORKFLOW.read_text())
        teacher = json.loads((root.parent / 'full-quality/baseline/plain-tokens.json').read_text())
        previous = lifecycle.BASE / 'performance-qwen21-latest-2026-11-16'
        original = json.loads((previous / 'original.json').read_text())
        pixels = np.asarray(Image.open(previous / 'original.png').convert('RGBA'))
        sequence = [('cold-baseline', True, True), ('cold-product', False, True)]
        for index, baseline in enumerate([True, False, False, True, True, False, False, True, True, False]):
            sequence.append((f'fixed-{index}', baseline, True))
        sequence += [('natural-baseline', True, False), ('natural-product', False, False),
                     ('seed-baseline', True, False), ('seed-product', False, False),
                     ('small', False, False), ('square', False, False), ('reload', False, False)]
        for label, baseline, fixed in sequence:
            current = copy.deepcopy(graph)
            if label.startswith('seed') or label == 'small':
                current['459:471']['inputs']['sampling_mode.seed'] += 1
                current['459:488']['inputs']['noise_seed'] += 1
            if label in ('small', 'square'):
                current['13']['inputs'].update(aspect_ratio='1:1 (Square)', megapixels=.25 if label == 'small' else 1.)
            if label == 'square':
                current['459:471']['inputs']['thinking'] = False
            if label == 'reload':
                request('/free', {'unload_models': True, 'free_memory': True})
                samples = []
                for _ in range(6):
                    time.sleep(10)
                    samples.append(request('/system_stats'))
                (api / 'unload.json').write_text(json.dumps(samples, indent=2))
            current['461']['inputs']['filename_prefix'] = 'gaudi/qwen80_pe/' + label
            extra = dict(qwen_baseline=baseline, qwen_teacher=teacher if fixed else None,
                         moss_recompute_nodes=['TextGenerate', 'TextEncodeQwenImage21', 'SamplerCustomAdvanced', 'VAEDecode'])
            pid = request('/prompt', {'prompt': current, 'extra_data': extra})['prompt_id']
            print(label, pid, flush=True)
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                assert server.poll() is None, 'Validation server exited during execution'
                history = request('/history/' + pid)
                if pid in history:
                    break
                time.sleep(.5)
            else:
                raise TimeoutError(pid)
            result = history[pid]
            (api / (label + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
            assert result['status']['status_str'] == 'success', result['status']
            cached = next(message[1]['nodes'] for message in result['status']['messages'] if message[0] == 'execution_cached')
            assert not {'459:471', '459:452', '459:486', '459:457'} & set(cached), cached
            item = result['outputs']['461']['images'][0]
            image = lifecycle.PRODUCT / 'output' / item['subfolder'] / item['filename']
            caption_equal = result['outputs']['459:472']['text'] == original['outputs']['459:472']['text']
            rgba_equal = np.array_equal(pixels, np.asarray(Image.open(image).convert('RGBA')))
            if fixed or label == 'natural-baseline':
                assert caption_equal and rgba_equal, (label, caption_equal, rgba_equal)
            stamps = [message[1]['timestamp'] for message in result['status']['messages']
                      if message[0] in ('execution_start', 'execution_success')]
            tokens = json.loads((api / (pid + '-tokens.json')).read_text())
            row = dict(label=label, baseline=baseline, fixed=fixed, prompt_id=pid,
                       seconds=(stamps[-1] - stamps[0]) / 1000, tokens=len(tokens),
                       image=str(image), caption_equal=caption_equal, rgba_equal=rgba_equal)
            results.append(row)
            (api / 'results.json').write_text(json.dumps(results, indent=2))
            print(row, flush=True)
        assert not request('/queue')['queue_running']
    finally:
        server.terminate()
        try:
            server.wait(timeout=40)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    root = args.root
    lifecycle.ROOT = root
    while args.wait_pid and psutil.pid_exists(args.wait_pid):
        time.sleep(2)
    lock = (lifecycle.BASE / '.qwen80-pe.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    files = json.loads((root / 'files.json').read_text())
    queue = lifecycle.request('/queue')
    assert not queue['queue_running'] and not queue['queue_pending'], queue
    backup = root / 'baseline-source'
    backup.mkdir()
    for name in files:
        target = backup / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lifecycle.PRODUCT / name, target)
    environment = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
                       PYTHONPATH=os.pathsep.join((str(lifecycle.PRODUCT), str(lifecycle.PROBES))))
    try:
        subprocess.run(['./stop-hpu.sh'], cwd=lifecycle.PRODUCT, check=True)
        for name in files:
            shutil.copy2(root / 'candidate-source' / name, lifecycle.PRODUCT / name)
        tests = ['generation_readback_test.py', 'hpu_qwen_generation_test.py', 'qwen35_decode_test.py',
                 'hpu_qwen_norm_test.py', 'hpu_qwen21_block_test.py', 'hpu_mlp_test.py',
                 'hpu_rope_test.py', 'hpu_kv_attention_test.py']
        lifecycle.run('tests', [sys.executable, '-m', 'pytest', '-q'] +
                      ['tests-unit/comfy_test/' + name for name in tests], environment)
        teacher = root.parent / 'full-quality/baseline/plain-tokens.json'
        lifecycle.run('product-logits', [sys.executable, str(lifecycle.PROBES / 'qwen21_text_profile.py'),
                      str(root / 'product-logits'), '--workflow', str(lifecycle.WORKFLOW),
                      '--teacher', str(teacher), '--quality-only', '--no-region-labels'], environment)
        workflows(root, environment)
        text_probe = [sys.executable, str(lifecycle.PROBES / 'qwen21_text_profile.py')]
        common = ['--workflow', str(lifecycle.WORKFLOW), '--teacher', str(teacher), '--no-region-labels']
        lifecycle.run('trace', text_probe + [str(root / 'trace'), '--window-tokens', '160',
                      '--window-steps', '64', '--profile-start-token', '64'] + common, environment)
        native_environment = lifecycle.native_environment('native-text', environment)
        lifecycle.run('native-text', text_probe + [str(root / 'native-text'), '--native',
                      '--window-tokens', '1632', '--window-steps', '64', '--profile-start-token', '1536'] + common,
                      native_environment)
        hashes = {name: hashlib.sha256((lifecycle.PRODUCT / name).read_bytes()).hexdigest() for name in files}
        (root / 'validated-hashes.json').write_text(json.dumps(hashes, indent=2))
    finally:
        for name in files:
            shutil.copy2(backup / name, lifecycle.PRODUCT / name)
        subprocess.run(['./start-hpu.sh'], cwd=lifecycle.PRODUCT, check=True, timeout=180)
        (root / 'restored-queue.json').write_text(json.dumps(lifecycle.request('/queue')))


if __name__ == '__main__':
    main()
