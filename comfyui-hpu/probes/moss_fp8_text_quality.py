"""Paired natural Qwen BF16/FP8 expansion, images and reload through the API."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.parse
import urllib.request

root = Path(os.environ['MOSS_SYSTEM_ROOT']) / os.environ.get('MOSS_QUALITY_DIRECTORY', 'quality')
root.mkdir(exist_ok=True)
probes = Path(__file__).parent
source = json.loads((root.parent.parent / 'performance-2026-11-11-fp8/source.json').read_text())
url = 'http://127.0.0.1:8190'

def api(path, data=None):
    request = urllib.request.Request(url + path, data=None if data is None else json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
        return json.loads(data) if data else None


def run(label, graph):
    prompt_id = api('/prompt', {'prompt': graph, 'extra_data': {'moss_diagnostic': True}})['prompt_id']
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
    request = [event for event in events if event.get('prompt_id') == prompt_id]
    checks = [event for event in request if event['kind'] == 'tensor_check']
    assert len(checks) >= 2 and all(event['finite'] for event in checks)
    end = next(event for event in request if event['kind'] == 'request_end')
    assert not end['fallback_counts'], end['fallback_counts']
    print(label, time.monotonic() - start, [event['output_tokens'] for event in request if event['kind'] == 'generate_end'], flush=True)


env = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so', PYTHONPATH=str(Path.cwd()), MOSS_EVENTS=str(root / 'events.jsonl'))
command = [sys.executable, str(probes / 'moss_observed_server.py'), '--disable-dynamic-vram', '--disable-pinned-memory', '--disable-async-offload', '--disable-xformers', '--use-pytorch-cross-attention', '--listen', '127.0.0.1', '--port', '8190']
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
        for mode, filename in ([] if os.environ.get('MOSS_QUALITY_TAIL') == '1' else [('bf16', 'qwen3.5_9b_bf16_from_int8.safetensors'), ('fp8', 'qwen3.5_9b_fp8_from_int8.safetensors')]):
            if os.environ.get('MOSS_QUALITY_MODE') and mode != os.environ['MOSS_QUALITY_MODE']:
                continue
            for label, prompt in [('teapot', 'A red ceramic teapot on a wooden table beside a sunlit window, realistic product photography.'),
                                  ('poster', 'A clean poster featuring a blue ceramic vase, three yellow flowers, a white background, and large readable text MOSS at the top.'),
                                  ('person', 'Portrait photo of an elderly woman wearing a red wool scarf, holding a white ceramic cup, natural skin texture, soft window light.')]:
                graph = copy.deepcopy(source)
                graph['18:2']['inputs']['clip_name'] = filename
                graph['18:17']['inputs']['value'] = prompt
                run(mode + '-' + label, graph)
        if os.environ.get('MOSS_QUALITY_TAIL') == '1':
            graph = copy.deepcopy(source)
            graph['18:2']['inputs']['clip_name'] = 'qwen3.5_9b_fp8_from_int8.safetensors'
            graph['18:11']['inputs']['max_length'] = 32
            run('fp8-warm-1024', graph)
        subprocess.run([sys.executable, str(probes / 'moss_memory_release.py'), str(root), url], check=True)
        graph['19']['inputs']['megapixels'] = .25
        graph['18:11']['inputs']['mtp'] = 'off'
        graph['18:11']['inputs']['max_length'] = 32
        run('fp8-reloaded-off-512', graph)
        graph['18:2']['inputs']['clip_name'] = 'qwen3.5_9b_int8_convrot.safetensors'
        graph['18:11']['inputs']['mtp'] = 'auto'
        run('int8-regression-512', graph)
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
