"""Profile the same warm negative DiT forward in baseline and product modes."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error

import qwen21_latest_jobs as lifecycle
from qwen80_validation_jobs import request


def measure(root, environment, history):
    graph = copy.deepcopy(history['prompt'][2])
    graph['459:452']['inputs']['prompt'] = history['outputs']['459:472']['text'][0]
    for node in ('459:471', '459:472', '459:473', '459:474', '459:475'):
        graph.pop(node, None)
    template = lifecycle.BASE / 'performance-qwen-hbm-2026-11-15/text0-c16383-b255-confirm/profiler.json'
    for variant in ('baseline', 'product'):
        directory = root / ('native-' + variant)
        directory.mkdir(parents=True)
        config = json.loads(template.read_text())
        config['GeneralSettings']['values']['outdir']['value'] = str(directory / 'hw')
        config['GeneralSettings']['values']['session']['value'] = 'd1-' + variant
        (directory / 'profiler.json').write_text(json.dumps(config, indent=2))
        env = dict(environment, QWEN_D1_ROOT=str(directory), QWEN_D1_PRODUCT='1',
                   MOSS_EVENTS=str(directory / 'events.jsonl'), HABANA_PROFILE='1',
                   HABANA_PROF_CONFIG=str(directory / 'profiler.json'))
        with (directory / 'server.log').open('w') as log:
            server = subprocess.Popen([sys.executable, '-c', 'import qwen_d1_server; qwen_d1_server.main()',
                '--disable-dynamic-vram', '--disable-pinned-memory', '--disable-async-offload',
                '--disable-xformers', '--use-pytorch-cross-attention', '--disable-all-custom-nodes',
                '--listen', '127.0.0.1', '--port', '8190'], cwd=lifecycle.PRODUCT,
                env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                for _ in range(120):
                    assert server.poll() is None, 'Hardware server exited'
                    try:
                        request('/queue')
                        break
                    except urllib.error.URLError:
                        time.sleep(1)
                else:
                    raise TimeoutError('Hardware startup')
                timings = []
                for label in ('cold', 'plain-0', 'plain-1', 'plain-2', 'native'):
                    current = copy.deepcopy(graph)
                    current['461']['inputs']['filename_prefix'] = f'gaudi/qwen_d1/hardware-{variant}-{label}'
                    mode = '' if label == 'cold' else 'native' if label == 'native' else 'plain'
                    extra = dict(d1_candidate=variant, d1_profile=mode,
                                 moss_recompute_nodes=['SamplerCustomAdvanced', 'VAEDecode'])
                    pid = request('/prompt', dict(prompt=current, extra_data=extra))['prompt_id']
                    deadline = time.monotonic() + 1200
                    while time.monotonic() < deadline:
                        assert server.poll() is None, 'Hardware server exited during request'
                        result = request('/history/' + pid)
                        if pid in result:
                            break
                        time.sleep(.5)
                    else:
                        raise TimeoutError(pid)
                    result = result[pid]
                    (directory / (label + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
                    assert result['status']['status_str'] == 'success', result['status']
                    if mode == 'plain':
                        timings.append(json.loads((directory / 'plain-window.json').read_text())['seconds'])
                        (directory / 'plain-windows.json').write_text(json.dumps(timings))
                    print('hardware', variant, label, pid, flush=True)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
