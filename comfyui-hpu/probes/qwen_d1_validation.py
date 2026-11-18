"""Interleave complete Qwen workflows for D1 precision candidates."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error

import qwen21_latest_jobs as lifecycle
from qwen80_validation_jobs import request


def workflows(root, environment, names):
    (root / 'capture').mkdir()
    environment = dict(environment, QWEN_D1_ROOT=str(root), MOSS_EVENTS=str(root / 'events.jsonl'),
                       QWEN_D1_CANDIDATES=','.join(names))
    results = []
    server_module = environment.get('QWEN_VALIDATION_SERVER', 'qwen_d1_server')
    with (root / 'server.log').open('w') as log:
        server = subprocess.Popen([sys.executable, '-c', f'import {server_module}; {server_module}.main()',
            '--disable-dynamic-vram', '--disable-pinned-memory', '--disable-async-offload',
            '--disable-xformers', '--use-pytorch-cross-attention', '--disable-all-custom-nodes',
            '--listen', '127.0.0.1', '--port', '8190'], cwd=lifecycle.PRODUCT,
            env=environment, stdout=log, stderr=subprocess.STDOUT)
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
            variants = ['baseline', *names]
            product = environment.get('QWEN_D1_PRODUCT') == '1'
            sequence = [('cold-' + name, name, 'cold') for name in variants]
            for repeat in range(5):
                order = variants if repeat % 2 == 0 else list(reversed(variants))
                sequence += [(f'warm-{repeat}-{name}', name, 'warm') for name in order]
            if product:
                for repeat in range(5):
                    order = variants if repeat % 2 == 0 else list(reversed(variants))
                    sequence += [(f'full-{repeat}-{name}', name, 'full') for name in order]
            for case in ('seed', 'size512', 'size1024', 'kv-off', 'trajectory', 'diagnostic'):
                sequence += [(case + '-' + name, name, case) for name in variants]
            if product:
                for case in ('blue', 'red', 'blue-return'):
                    sequence += [(case + '-' + name, name, case) for name in variants]
                sequence.append(('reload-product', 'product', 'reload'))
            for label, name, case in sequence:
                current = copy.deepcopy(graph)
                if case == 'seed':
                    current['459:488']['inputs']['noise_seed'] += 1
                if case in ('size512', 'size1024'):
                    current['13']['inputs'].update(aspect_ratio='1:1 (Square)', megapixels=.25 if case == 'size512' else 1.)
                if case == 'kv-off':
                    current['459:490']['inputs']['device'] = 'off'
                if case in ('blue', 'red', 'blue-return'):
                    current['459:471']['inputs']['prompt'] = current['459:471']['inputs']['prompt'].replace(
                        '灰褐色', '深红色' if case == 'red' else '深蓝色')
                if case == 'reload':
                    request('/free', dict(unload_models=True, free_memory=True))
                    samples = []
                    for _ in range(6):
                        time.sleep(10)
                        samples.append(request('/system_stats'))
                    (root / 'unload.json').write_text(json.dumps(samples, indent=2))
                current['461']['inputs']['filename_prefix'] = 'gaudi/qwen_d1/' + label
                extra = dict(d1_candidate=name, d1_capture=label == 'cold-baseline' and not product,
                             d1_capture_all_layers=label == 'cold-baseline',
                             d1_trajectory=case == 'trajectory', d1_diagnostic=case == 'diagnostic',
                             moss_recompute_nodes=['SamplerCustomAdvanced', 'VAEDecode'])
                if case in ('full', 'blue', 'red', 'blue-return', 'reload'):
                    extra['moss_recompute_nodes'] += ['TextGenerate', 'TextEncodeQwenImage21']
                pid = request('/prompt', dict(prompt=current, extra_data=extra))['prompt_id']
                deadline = time.monotonic() + 1200
                while time.monotonic() < deadline:
                    assert server.poll() is None, 'Validation server exited during request'
                    history = request('/history/' + pid)
                    if pid in history:
                        break
                    time.sleep(.5)
                else:
                    raise TimeoutError(pid)
                result = history[pid]
                (root / (label + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
                assert result['status']['status_str'] == 'success', result['status']
                cached = next(row[1]['nodes'] for row in result['status']['messages'] if row[0] == 'execution_cached')
                assert not {'459:486', '459:457'} & set(cached), cached
                stamps = [row[1]['timestamp'] for row in result['status']['messages']
                          if row[0] in ('execution_start', 'execution_success')]
                row = dict(label=label, candidate=name, prompt_id=pid, seconds=(stamps[-1] - stamps[0]) / 1000)
                results.append(row)
                (root / 'results.json').write_text(json.dumps(results, indent=2))
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
    parser.add_argument('--names', nargs='+', default=['split-fast', 'center-split-fast'])
    args = parser.parse_args()
    root = args.root
    root.mkdir()
    lifecycle.ROOT = root
    lock = (lifecycle.BASE / '.qwen80-pe.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    queue = lifecycle.request('/queue')
    assert not queue['queue_running'] and not queue['queue_pending'], queue
    environment = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
                       PYTHONPATH=os.pathsep.join((str(lifecycle.PRODUCT), str(lifecycle.PROBES))))
    try:
        subprocess.run(['./stop-hpu.sh'], cwd=lifecycle.PRODUCT, check=True)
        workflows(root, environment, args.names)
    finally:
        subprocess.run(['./start-hpu.sh'], cwd=lifecycle.PRODUCT, check=True, timeout=180)
        lifecycle.verify_restoration()


if __name__ == '__main__':
    main()
