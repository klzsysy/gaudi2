"""Verify the unstaged product candidate through normal Qwen compute nodes."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error

import qwen21_latest_jobs as lifecycle
from qwen80_validation_jobs import request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    options = parser.parse_args()
    root = options.root
    root.mkdir(exist_ok=True)
    environment = dict(os.environ, MOSS_EVENTS=str(root / 'events.jsonl'))
    graph = json.loads(lifecycle.WORKFLOW.read_text())
    with (root / 'server.log').open('w') as log:
        server = subprocess.Popen([sys.executable, str(lifecycle.PROBES / 'moss_observed_server.py'),
            '--disable-dynamic-vram', '--disable-pinned-memory', '--disable-async-offload',
            '--disable-xformers', '--use-pytorch-cross-attention', '--disable-all-custom-nodes',
            '--listen', '127.0.0.1', '--port', '8190'], cwd=lifecycle.PRODUCT,
            env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(120):
                assert server.poll() is None, 'Product server exited'
                try:
                    request('/queue')
                    break
                except urllib.error.URLError:
                    time.sleep(1)
            else:
                raise TimeoutError('Product startup')
            results = []
            cases = ['cold', 'warm-0', 'warm-1', 'warm-2', 'full-0', 'full-1', 'full-2',
                     'size512', 'size1024', 'seed', 'kv-off', 'blue', 'red', 'blue-return', 'original', 'reload']
            for case in cases:
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
                current['461']['inputs']['filename_prefix'] = 'gaudi/qwen_d2/product-' + case
                nodes = ['SamplerCustomAdvanced', 'VAEDecode']
                if case.startswith('full') or case in ('blue', 'red', 'blue-return', 'original', 'reload'):
                    nodes += ['TextGenerate', 'TextEncodeQwenImage21']
                pid = request('/prompt', dict(prompt=current, extra_data=dict(moss_recompute_nodes=nodes)))['prompt_id']
                deadline = time.monotonic()+900
                while time.monotonic() < deadline:
                    assert server.poll() is None, 'Product exited during request'
                    response = request('/history/' + pid)
                    if pid in response:
                        break
                    time.sleep(.5)
                else:
                    raise TimeoutError(pid)
                result = response[pid]
                (root / (case + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
                assert result['status']['status_str'] == 'success', result['status']
                cached = next(message[1]['nodes'] for message in result['status']['messages'] if message[0] == 'execution_cached')
                assert not {'459:486', '459:457'} & set(cached), cached
                if 'TextGenerate' in nodes:
                    assert '459:471' not in cached
                stamps = [message[1]['timestamp'] for message in result['status']['messages']
                          if message[0] in ('execution_start', 'execution_success')]
                row = dict(case=case, prompt_id=pid, seconds=(stamps[-1]-stamps[0])/1000)
                results.append(row)
                (root / 'results.json').write_text(json.dumps(results, indent=2))
                print(json.dumps(row), flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=40)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == '__main__':
    main()
