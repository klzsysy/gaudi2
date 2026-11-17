"""Validate a source snapshot across edited prompts, warm repeats and reload."""
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

import qwen21_latest_jobs as lifecycle
from qwen80_validation_jobs import request


def workflows(root, environment):
    api = root / 'api'
    api.mkdir()
    environment = dict(environment, QWEN_STARTUP_VALIDATION=str(root), MOSS_EVENTS=str(api / 'events.jsonl'))
    history = json.loads((lifecycle.BASE / 'performance-qwen-startup-2026-11-16/history-before.json').read_text())
    graphs = [value['prompt'][2] for value in history.values()]
    references = {}
    with (api / 'server.log').open('w') as log:
        server = subprocess.Popen([sys.executable, '-c', 'import qwen_startup_server; qwen_startup_server.main()',
            '--disable-dynamic-vram', '--disable-pinned-memory', '--disable-async-offload',
            '--disable-xformers', '--use-pytorch-cross-attention', '--disable-all-custom-nodes',
            '--listen', '127.0.0.1', '--port', '8190'], cwd=lifecycle.PRODUCT,
            env=environment, stdout=log, stderr=subprocess.STDOUT)
        results = []
        try:
            for _ in range(120):
                assert server.poll() is None
                try:
                    request('/queue')
                    break
                except urllib.error.URLError:
                    time.sleep(1)
            else:
                raise TimeoutError('Startup')
            sequence = [('cold-baseline', True, 0), ('cold-product', False, 0)]
            for index, graph_index in enumerate([-6, -5, -4, -1]):
                sequence += [(f'edit-{index}-baseline', True, graph_index), (f'edit-{index}-product', False, graph_index)]
            for index, baseline in enumerate([True, False, False, True, True, False, False, True, True, False]):
                sequence.append((f'warm-{index}', baseline, 0))
            sequence += [('reload', False, 0), ('reload-edit', False, -1)]
            for label, baseline, graph_index in sequence:
                graph = copy.deepcopy(graphs[graph_index])
                graph['461']['inputs']['filename_prefix'] = 'gaudi/qwen_startup/' + label
                if label == 'reload':
                    request('/free', {'unload_models': True, 'free_memory': True})
                    samples = []
                    for _ in range(6):
                        time.sleep(10)
                        samples.append(request('/system_stats'))
                    (api / 'unload.json').write_text(json.dumps(samples, indent=2))
                extra = dict(qwen_baseline=baseline, qwen_measure_startup=not label.startswith('warm'), moss_recompute_nodes=[
                    'TextGenerate', 'TextEncodeQwenImage21', 'SamplerCustomAdvanced', 'VAEDecode'])
                pid = request('/prompt', {'prompt': graph, 'extra_data': extra})['prompt_id']
                print(label, pid, flush=True)
                deadline = time.monotonic() + 1200
                while time.monotonic() < deadline:
                    assert server.poll() is None
                    history = request('/history/' + pid)
                    if pid in history:
                        break
                    time.sleep(.5)
                else:
                    raise TimeoutError(pid)
                result = history[pid]
                (api / (label + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
                assert result['status']['status_str'] == 'success', result['status']
                cached = next(m[1]['nodes'] for m in result['status']['messages'] if m[0] == 'execution_cached')
                assert not {'459:471', '459:452', '459:486', '459:457'} & set(cached), cached
                item = result['outputs']['461']['images'][0]
                path = lifecycle.PRODUCT / 'output' / item['subfolder'] / item['filename']
                pixels = np.asarray(Image.open(path).convert('RGBA'))
                caption = result['outputs']['459:472']['text']
                generation = json.loads((api / (pid + '-generation.json')).read_text())
                if baseline:
                    references[graph_index] = (pixels, caption, generation['tokens'])
                reference = references[graph_index]
                stamps = [m[1]['timestamp'] for m in result['status']['messages'] if m[0] in ('execution_start', 'execution_success')]
                row = dict(label=label, baseline=baseline, graph_index=graph_index, prompt_id=pid,
                    seconds=(stamps[-1] - stamps[0]) / 1000, text_seconds=generation['seconds'],
                    sample_times=generation['sample_times'], compilations=generation['compilations'],
                    tokens=len(generation['tokens']), tokens_equal=reference[2] == generation['tokens'],
                    caption_equal=reference[1] == caption, rgba_equal=np.array_equal(reference[0], pixels), image=str(path))
                results.append(row)
                (api / 'results.json').write_text(json.dumps(results, indent=2))
                print(row, flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=40)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--logits-only', action='store_true')
    arguments = parser.parse_args()
    root = arguments.root
    lifecycle.ROOT = root
    files = json.loads((root / 'files.json').read_text())
    lock = (lifecycle.BASE / '.qwen80-pe.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    queue = lifecycle.request('/queue')
    assert not queue['queue_running'] and not queue['queue_pending'], queue
    backup = root / 'baseline-source'
    backup.mkdir()
    for name in files:
        path = backup / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lifecycle.PRODUCT / name, path)
    environment = dict(os.environ, PT_HPU_LAZY_MODE='0',
        GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so', TORCH_LOGS='recompiles',
        PYTHONPATH=os.pathsep.join((str(lifecycle.PRODUCT), str(lifecycle.PROBES))))
    try:
        subprocess.run(['./stop-hpu.sh'], cwd=lifecycle.PRODUCT, check=True)
        for name in files:
            shutil.copy2(root / 'candidate-source' / name, lifecycle.PRODUCT / name)
        tests = ['generation_readback_test.py', 'hpu_qwen_generation_test.py', 'qwen35_decode_test.py',
                 'hpu_qwen_norm_test.py', 'hpu_qwen21_block_test.py', 'hpu_mlp_test.py',
                 'hpu_rope_test.py', 'hpu_kv_attention_test.py']
        if arguments.logits_only:
            previous = lifecycle.BASE / 'performance-qwen80-pe-2026-11-16'
            lifecycle.run('logits', [sys.executable, str(lifecycle.PROBES / 'qwen21_text_profile.py'),
                str(root / 'logits'), '--workflow', str(lifecycle.WORKFLOW), '--quality-only',
                '--teacher', str(previous / 'full-quality/baseline/plain-tokens.json'), '--no-region-labels'], environment)
            lifecycle.run('compare', [sys.executable, str(lifecycle.PROBES / 'qwen80_compare_logits.py'),
                str(previous / 'product-validation4/product-logits/logits.pt'), str(root / 'logits/logits.pt'),
                '--tokens', '1909', '--output', str(root / 'logits-comparison.json')], environment)
        else:
            lifecycle.run('tests', [sys.executable, '-m', 'pytest', '-q'] +
                          ['tests-unit/comfy_test/' + name for name in tests], environment)
            workflows(root, environment)
        hashes = {name: hashlib.sha256((lifecycle.PRODUCT / name).read_bytes()).hexdigest() for name in files}
        (root / 'validated-hashes.json').write_text(json.dumps(hashes, indent=2))
    finally:
        for name in files:
            shutil.copy2(backup / name, lifecycle.PRODUCT / name)
        subprocess.run(['./start-hpu.sh'], cwd=lifecycle.PRODUCT, check=True, timeout=180)
        (root / 'restored-queue.json').write_text(json.dumps(lifecycle.request('/queue')))


if __name__ == '__main__':
    main()
