"""Run serialized D1 capture and screening jobs, restoring ordinary serving."""
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


def capture(root, environment):
    (root / 'capture').mkdir()
    environment = dict(environment, QWEN_D1_ROOT=str(root), MOSS_EVENTS=str(root / 'events.jsonl'))
    results = []
    with (root / 'server.log').open('w') as log:
        server = subprocess.Popen([sys.executable, '-c', 'import qwen_d1_server; qwen_d1_server.main()',
            '--disable-dynamic-vram', '--disable-pinned-memory', '--disable-async-offload',
            '--disable-xformers', '--use-pytorch-cross-attention', '--disable-all-custom-nodes',
            '--listen', '127.0.0.1', '--port', '8190'], cwd=lifecycle.PRODUCT,
            env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(120):
                assert server.poll() is None, 'Capture server exited'
                try:
                    request('/queue')
                    break
                except urllib.error.URLError:
                    time.sleep(1)
            else:
                raise TimeoutError('Capture startup')
            graph = json.loads(lifecycle.WORKFLOW.read_text())
            for label in ['capture', *[f'warm-{i}' for i in range(5)], 'diagnostic']:
                current = copy.deepcopy(graph)
                current['461']['inputs']['filename_prefix'] = 'gaudi/qwen_d1/' + label
                extra = dict(d1_capture=label == 'capture', d1_diagnostic=label == 'diagnostic',
                             moss_recompute_nodes=['SamplerCustomAdvanced', 'VAEDecode'])
                pid = request('/prompt', dict(prompt=current, extra_data=extra))['prompt_id']
                deadline = time.monotonic() + 1200
                while time.monotonic() < deadline:
                    assert server.poll() is None, 'Capture server exited during request'
                    history = request('/history/' + pid)
                    if pid in history:
                        break
                    time.sleep(.5)
                else:
                    raise TimeoutError(pid)
                result = history[pid]
                (root / (label + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
                assert result['status']['status_str'] == 'success', result['status']
                stamps = [row[1]['timestamp'] for row in result['status']['messages']
                          if row[0] in ('execution_start', 'execution_success')]
                row = dict(label=label, prompt_id=pid, seconds=(stamps[-1] - stamps[0]) / 1000)
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
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--screen', action='store_true')
    parser.add_argument('--screen-names', nargs='+')
    parser.add_argument('--all-inputs', action='store_true')
    parser.add_argument('--screen-label', default='screen')
    args = parser.parse_args()
    root = args.root
    root.mkdir(exist_ok=True)
    lifecycle.ROOT = root
    lock = (lifecycle.BASE / '.qwen80-pe.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    queue = lifecycle.request('/queue')
    assert not queue['queue_running'] and not queue['queue_pending'], queue
    environment = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
                       PYTHONPATH=os.pathsep.join((str(lifecycle.PRODUCT), str(lifecycle.PROBES))))
    try:
        subprocess.run(['./stop-hpu.sh'], cwd=lifecycle.PRODUCT, check=True)
        if args.capture:
            capture(root, environment)
        if args.screen:
            command = [sys.executable, str(lifecycle.PROBES / 'qwen_d1_screen.py'), str(root), '--output', args.screen_label]
            if args.screen_names:
                command += ['--names', *args.screen_names]
            if args.all_inputs:
                command.append('--all-inputs')
            lifecycle.run(args.screen_label, command, environment)
    finally:
        subprocess.run(['./start-hpu.sh'], cwd=lifecycle.PRODUCT, check=True, timeout=180)
        lifecycle.verify_restoration()


if __name__ == '__main__':
    main()
