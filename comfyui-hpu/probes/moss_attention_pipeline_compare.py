"""Paired long-image attention experiment with fresh same-seed A/A images."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.parse
import urllib.request


parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('--history', type=Path, required=True)
options = parser.parse_args()
root = options.root
root.mkdir(parents=True, exist_ok=True)
history = json.loads(options.history.read_text())
source = list(history.values())[-1]
graph = copy.deepcopy(source['prompt'][2])
graph['18:4']['inputs']['text'] = source['outputs']['18:13']['text'][0]
graph.pop('18:13')
(root / 'source.json').write_text(json.dumps(graph, indent=2))
url = 'http://127.0.0.1:8190'
records = []


def api(route, data=None):
    request = urllib.request.Request(url + route, data=None if data is None else json.dumps(data).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read()
        return json.loads(body) if body else None


def run(label, mode, seed, width=1088, height=1936, diagnostic=False):
    prompt = copy.deepcopy(graph)
    prompt['18:6']['inputs'].update(width=width, height=height)
    prompt['18:7']['inputs']['seed'] = seed
    extra = dict(moss_recompute=True, moss_recompute_nodes=['KSampler', 'VAEDecode'],
                 attention_dataflow=mode, moss_diagnostic=diagnostic)
    prompt_id = api('/prompt', {'prompt': prompt, 'extra_data': extra})['prompt_id']
    start = time.monotonic()
    while prompt_id not in (history := api('/history/' + prompt_id)):
        if time.monotonic() - start > 1200:
            raise TimeoutError(prompt_id)
        time.sleep(.25)
    result = history[prompt_id]
    assert result['status']['status_str'] == 'success', result['status']
    (root / f'{label}-history.json').write_text(json.dumps(result))
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    events = [e for e in events if e.get('prompt_id') == prompt_id]
    begin = next(e for e in events if e['kind'] == 'request_start')
    end = next(e for e in events if e['kind'] == 'request_end')
    assert not end['fallback_counts'], end['fallback_counts']
    starts, times = {}, {}
    for event in events:
        if event['kind'] == 'node_start':
            starts[event['node']] = event['mono_ns']
        elif event['kind'] == 'node_end':
            name = event['class_type']
            times[name] = times.get(name, 0) + (event['mono_ns'] - starts[event['node']]) / 1e9
    record = dict(label=label, mode=mode, seed=seed, width=width, height=height,
                  seconds=(end['mono_ns'] - begin['mono_ns']) / 1e9, nodes=times,
                  fallback=end['fallback_counts'], memory=end['memory_stats'])
    records.append(record)
    (root / 'runs.json').write_text(json.dumps(records, indent=2))
    for output in result['outputs'].values():
        for image in output.get('images', []):
            with urllib.request.urlopen(url + '/view?' + urllib.parse.urlencode(image)) as response:
                (root / f'{label}.png').write_bytes(response.read())
    print(json.dumps({k: v for k, v in record.items() if k != 'memory'}), flush=True)


env = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
           PYTHONPATH=str(Path.cwd()), MOSS_EVENTS=str(root / 'events.jsonl'), MOSS_ATTENTION_DATAFLOW_AB='1')
command = [sys.executable, str(Path(__file__).with_name('moss_three_stage_server.py')),
           '--disable-dynamic-vram', '--disable-pinned-memory', '--disable-async-offload',
           '--disable-xformers', '--use-pytorch-cross-attention', '--disable-all-custom-nodes',
           '--listen', '127.0.0.1', '--port', '8190']
with (root / 'server.log').open('w') as log:
    server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            if server.poll() is not None:
                raise RuntimeError('Experiment server exited')
            try:
                api('/system_stats')
                break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError('Startup')
        for mode in ('baseline', 'long-query'):
            run(f'{mode}-cold', mode, 734670000)
        for trial, offset in enumerate((0, 0, 1, 2, 0)):
            modes = ('baseline', 'long-query') if trial % 2 == 0 else ('long-query', 'baseline')
            for mode in modes:
                run(f'{mode}-warm-{trial}', mode, 734670000 + offset)
        for mode in ('baseline', 'long-query'):
            run(f'{mode}-smaller', mode, 734670000, 1024, 1824)
        run('long-query-diagnostic', 'long-query', 734670000, diagnostic=True)
        release = root / 'release'
        release.mkdir(exist_ok=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name('moss_memory_release.py')), str(release), url], check=True)
        run('long-query-reload-512', 'long-query', 734670000, 512, 512, diagnostic=True)
        assert not any(api('/queue').values())
    finally:
        server.terminate()
        server.wait(timeout=60)
