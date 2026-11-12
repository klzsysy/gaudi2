"""Compare full recomputation and ordinary node-cache reuse on one workflow."""
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request


directory = Path(os.environ['MOSS_PIPELINE_DIR'])
directory.mkdir(parents=True, exist_ok=True)
source = json.loads((directory / 'source.json').read_text())
checkout = Path.cwd()
url = 'http://127.0.0.1:8190'


def api(route, data=None):
    request = urllib.request.Request(url + route, data=None if data is None else json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def run(label, graph, recompute, diagnostic=False):
    start = time.perf_counter()
    extra = {'moss_recompute': recompute, 'moss_diagnostic': diagnostic, 'moss_breakdown': diagnostic}
    prompt_id = api('/prompt', {'prompt': graph, 'extra_data': extra})['prompt_id']
    while prompt_id not in (history := api('/history/' + prompt_id)):
        if time.perf_counter() - start > 1200:
            raise TimeoutError(prompt_id)
        time.sleep(.2)
    history = history[prompt_id]
    (directory / (label + '-history.json')).write_text(json.dumps(history))
    assert history['status']['status_str'] == 'success', history['status']
    relevant = [json.loads(s) for s in (directory / 'events.jsonl').read_text().splitlines()]
    relevant = [e for e in relevant if e.get('prompt_id') == prompt_id]
    begin = next(e for e in relevant if e['kind'] == 'request_start')
    end = next(e for e in relevant if e['kind'] == 'request_end')
    assert not end['fallback_counts'], end['fallback_counts']
    cached = next((data['nodes'] for kind, data in history['status']['messages'] if kind == 'execution_cached'), [])
    generated = [e['output_tokens'] for e in relevant if e['kind'] == 'generate_end']
    record = {'label': label, 'prompt_id': prompt_id, 'request_seconds': (end['mono_ns'] - begin['mono_ns']) / 1e9,
              'client_seconds': time.perf_counter() - start, 'diagnostic': diagnostic, 'recompute': recompute,
              'cached_nodes': cached, 'output_tokens': generated[0] if generated else None}
    with (directory / 'runs.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    if label.startswith('cached-warm'):
        assert '18:11' in cached and not generated, record
    if recompute:
        assert generated, record
    print(json.dumps(record), flush=True)


env = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH=os.environ.get('GC_KERNEL_PATH', '/usr/lib/habanalabs/libtpc_kernels.so'),
           PYTHONPATH=str(checkout), MOSS_EVENTS=str(directory / 'events.jsonl'), MOSS_PIPELINE_DETAIL='1')
command = [sys.executable, str(Path(__file__).with_name(os.environ.get('MOSS_SERVER_SCRIPT', 'moss_observed_server.py'))), '--disable-dynamic-vram',
           '--disable-pinned-memory', '--disable-async-offload', '--disable-xformers', '--use-pytorch-cross-attention',
           '--listen', '127.0.0.1', '--port', '8190']
with (directory / 'server.log').open('w') as log, (directory / 'hpu.csv').open('w') as telemetry_log:
    server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    telemetry = None
    try:
        for _ in range(120):
            if server.poll() is not None:
                raise RuntimeError('Measurement server exited')
            try:
                api('/system_stats')
                break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError('Measurement server startup')
        telemetry = subprocess.Popen(['hl-smi', '--query-aip=timestamp,utilization.aip,utilization.memory,memory.used,power.draw,temperature.aip,clocks.current.soc,clocks.limit.soc,clocks.limit.tpc', '--format=csv,nounits', '-l', '100ms'], stdout=telemetry_log)
        for index in range(int(os.environ.get('MOSS_PIPELINE_FULL_RUNS','6'))):
            run('full-cold' if index == 0 else f'full-warm-{index}', source, True)
        for index in range(int(os.environ.get('MOSS_PIPELINE_CACHE_RUNS', '6'))):
            graph = copy.deepcopy(source)
            graph['18:7']['inputs']['seed'] = (source['18:7']['inputs']['seed'] + index + 1) % (2**64)
            run('cached-transition' if index == 0 else f'cached-warm-{index}', graph, False)
        if os.environ.get('MOSS_PIPELINE_DIAGNOSTIC','1') == '1':
            run('full-diagnostic', source, True, diagnostic=True)
        if os.environ.get('MOSS_BF16_ACCEPTANCE') == '1':
            subprocess.run([sys.executable, str(Path(__file__).with_name('moss_bf16_quality.py')), str(directory)], check=True)
        if os.environ.get('MOSS_TREE_QUALITY') == '1':
            subprocess.run([sys.executable, str(Path(__file__).with_name('moss_tree_quality.py')), str(directory)], check=True)
        if os.environ.get('MOSS_RELEASE') == '1':
            release = directory / 'release'
            release.mkdir(exist_ok=True)
            subprocess.run([sys.executable, str(Path(__file__).with_name('moss_memory_release.py')), str(release), url], check=True)
    finally:
        if telemetry is not None:
            telemetry.terminate()
            telemetry.wait(timeout=10)
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=60)
