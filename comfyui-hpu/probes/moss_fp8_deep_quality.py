"""Paired images with the same generated caption, seed and sampler settings."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.parse
import urllib.request

root = Path(os.environ['MOSS_DEEP_ROOT']) / os.environ.get('MOSS_DEEP_QUALITY', 'quality-attention')
root.mkdir(parents=True, exist_ok=True)
source = json.loads((root.parent / 'baseline/source.json').read_text())
url = 'http://127.0.0.1:8190'

def api(route, data=None):
    request = urllib.request.Request(url + route, data=None if data is None else json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        content = response.read()
        return json.loads(content) if content else None

def run(label, graph, candidate):
    extra = dict(moss_diagnostic=True, moss_fp8_attention=candidate,
                 moss_recompute_nodes=['KSampler', 'VAEDecode'])
    prompt_id = api('/prompt', dict(prompt=graph, extra_data=extra))['prompt_id']
    start = time.monotonic()
    while prompt_id not in (history := api('/history/' + prompt_id)):
        if time.monotonic() - start > 900:
            raise TimeoutError(prompt_id)
        time.sleep(.3)
    history = history[prompt_id]
    (root / (label + '-history.json')).write_text(json.dumps(history, indent=2))
    assert history['status']['status_str'] == 'success', history['status']
    for output in history['outputs'].values():
        for image in output.get('images', []):
            with urllib.request.urlopen(url + '/view?' + urllib.parse.urlencode(image)) as response:
                (root / (label + '.png')).write_bytes(response.read())
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    events = [event for event in events if event.get('prompt_id') == prompt_id]
    checks = [event for event in events if event['kind'] == 'tensor_check']
    assert len(checks) >= 2 and all(event['finite'] for event in checks)
    end = next(event for event in events if event['kind'] == 'request_end')
    assert not end['fallback_counts']
    if candidate:
        assert not any(event['kind'] == 'generate_start' for event in events)
    print(label, time.monotonic() - start, flush=True)

env = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so', PYTHONPATH=str(Path.cwd()),
           MOSS_PIPELINE_DIR=str(root), MOSS_EVENTS=str(root / 'events.jsonl'))
env['MOSS_DEEP_PRODUCT_AB' if os.environ.get('MOSS_DEEP_PRODUCT_AB') == '1' else 'MOSS_DEEP_ATTENTION_AB'] = '1'
command = [sys.executable, str(Path(__file__).with_name('moss_fp8_deep_server.py')), '--disable-dynamic-vram',
           '--disable-pinned-memory', '--disable-async-offload', '--disable-xformers', '--use-pytorch-cross-attention',
           '--listen', '127.0.0.1', '--port', '8190']
with (root / 'server.log').open('w') as log:
    server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            if server.poll() is not None:
                raise RuntimeError('Server exited')
            try:
                api('/system_stats')
                break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError('Startup')
        for label, prompt in [('teapot', 'A red ceramic teapot on a wooden table beside a sunlit window, realistic product photography.'),
                              ('poster', 'A clean poster featuring a blue ceramic vase, three yellow flowers, a white background, and large readable text MOSS at the top.'),
                              ('person', 'Portrait photo of an elderly woman wearing a red wool scarf, holding a white ceramic cup, natural skin texture, soft window light.')]:
            graph = copy.deepcopy(source)
            graph['18:17']['inputs']['value'] = prompt
            run(label + '-baseline', graph, False)
            run(label + '-candidate', graph, True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
