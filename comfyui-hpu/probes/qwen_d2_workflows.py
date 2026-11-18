"""Interleave fixed-caption D2 candidates through the ordinary compute nodes."""
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
    parser.add_argument('--modes', nargs='+', default=['baseline', 'block', 'loop', 'block-loop'])
    parser.add_argument('--rounds', type=int, default=5)
    options = parser.parse_args()
    root = options.root
    environment = dict(os.environ, QWEN_D1_ROOT=str(root), QWEN_D1_PRODUCT='1',
                       MOSS_EVENTS=str(root / 'events.jsonl'))
    history = json.loads((lifecycle.BASE / 'performance-qwen-d1-2026-11-17/product3/api/full-0-product.json').read_text())
    graph = copy.deepcopy(history['prompt'][2])
    graph['459:452']['inputs']['prompt'] = history['outputs']['459:472']['text'][0]
    for node in ('459:471', '459:472', '459:473', '459:474', '459:475'):
        graph.pop(node, None)
    with (root / 'server.log').open('w') as log:
        server = subprocess.Popen([sys.executable, '-c', 'import qwen_d2_server; qwen_d2_server.main()',
            '--disable-dynamic-vram', '--disable-pinned-memory', '--disable-async-offload',
            '--disable-xformers', '--use-pytorch-cross-attention', '--disable-all-custom-nodes',
            '--listen', '127.0.0.1', '--port', '8190'], cwd=lifecycle.PRODUCT,
            env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(120):
                assert server.poll() is None, 'D2 server exited'
                try:
                    request('/queue')
                    break
                except urllib.error.URLError:
                    time.sleep(1)
            else:
                raise TimeoutError('D2 startup')
            results = []
            cases = [('cold-' + mode, mode) for mode in options.modes]
            for index in range(options.rounds):
                order = options.modes if index % 2 == 0 else list(reversed(options.modes))
                cases.extend((f'warm-{index}-' + mode, mode) for mode in order)
            for label, mode in cases:
                graph['461']['inputs']['filename_prefix'] = 'gaudi/qwen_d2/' + label
                extra = dict(d1_candidate='product', d2_candidate=mode,
                             moss_recompute_nodes=['SamplerCustomAdvanced', 'VAEDecode'])
                pid = request('/prompt', dict(prompt=graph, extra_data=extra))['prompt_id']
                deadline = time.monotonic() + 900
                while time.monotonic() < deadline:
                    assert server.poll() is None, 'D2 server exited during request'
                    response = request('/history/' + pid)
                    if pid in response:
                        break
                    time.sleep(.5)
                else:
                    raise TimeoutError(pid)
                result = response[pid]
                (root / (label + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
                assert result['status']['status_str'] == 'success', result['status']
                stamps = [message[1]['timestamp'] for message in result['status']['messages']
                          if message[0] in ('execution_start', 'execution_success')]
                row = dict(label=label, candidate=mode, prompt_id=pid, seconds=(stamps[-1]-stamps[0])/1000)
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
