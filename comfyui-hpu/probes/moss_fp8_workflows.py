"""Run fixed-input MOSS FP8 workflows with node timing and finite/fallback checks."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import urllib.parse
root = Path(os.environ['MOSS_FP8_ROOT'])
root.mkdir(exist_ok=True, parents=True)
probes = Path(__file__).parent
source = json.loads((root / 'source.json').read_text())
url = 'http://127.0.0.1:8190'

def api(path, data=None):
    r = urllib.request.Request(url + path, data=None if data is None else json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=30) as f:
        b = f.read()
        return json.loads(b) if b else None
env = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so', PYTHONPATH=str(Path.cwd()), MOSS_EVENTS=str(root / 'events.jsonl'), MOSS_PIPELINE_DETAIL='1')
command = [sys.executable, str(probes / 'moss_fp8_server.py'), '--disable-dynamic-vram', '--disable-pinned-memory', '--disable-async-offload', '--disable-xformers', '--use-pytorch-cross-attention', '--listen', '127.0.0.1', '--port', '8190']
with (root / 'server.log').open('w') as log:
    server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            if server.poll() is not None:
                raise RuntimeError('server exited')
            try:
                api('/system_stats')
                break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError('startup')
        for i in range(int(os.environ.get('MOSS_FP8_RUNS', '4'))):
            graph = copy.deepcopy(source)
            extra = {'moss_diagnostic': i == 0}
            if os.environ.get('MOSS_FP8_FULL') == '1':
                graph['18:7']['inputs']['seed'] += i
            else:
                extra['moss_recompute_nodes'] = ['KSampler', 'VAEDecode']
            pid = api('/prompt', {'prompt': graph, 'extra_data': extra})['prompt_id']
            start = time.monotonic()
            while pid not in (h := api('/history/' + pid)):
                if time.monotonic() - start > 1800:
                    raise TimeoutError(pid)
                time.sleep(0.3)
            h = h[pid]
            (root / f'run-{i}-history.json').write_text(json.dumps(h))
            assert h['status']['status_str'] == 'success', h['status']
            for output in h['outputs'].values():
                for image in output.get('images', []):
                    with urllib.request.urlopen(url + '/view?' + urllib.parse.urlencode(image)) as response:
                        (root / f'image-{i}.png').write_bytes(response.read())
            print('completed', i, time.monotonic() - start, pid, flush=True)
        if os.environ.get('MOSS_FP8_ACCEPTANCE') == '1':
            for label, mp, prompt in [('poster', 0.5625, 'A clean poster featuring a blue ceramic vase, three yellow flowers, a white background, and large readable text MOSS at the top.'), ('person', 1.0, 'Portrait photo of an elderly woman wearing a red wool scarf, holding a white ceramic cup, natural skin texture, soft window light.'), ('reload', 0.25, 'A red ceramic teapot on a wooden table beside a sunlit window, realistic product photography.')]:
                if label == 'reload':
                    subprocess.run([sys.executable, str(probes / 'moss_memory_release.py'), str(root), url], check=True)
                graph = copy.deepcopy(source)
                graph['19']['inputs']['megapixels'] = mp
                graph['18:17']['inputs']['value'] = prompt
                pid = api('/prompt', {'prompt': graph, 'extra_data': {'moss_diagnostic': True}})['prompt_id']
                start = time.monotonic()
                while pid not in (h := api('/history/' + pid)):
                    if time.monotonic() - start > 900:
                        raise TimeoutError(pid)
                    time.sleep(0.3)
                h = h[pid]
                (root / (label + '-history.json')).write_text(json.dumps(h))
                assert h['status']['status_str'] == 'success', h['status']
                for output in h['outputs'].values():
                    for image in output.get('images', []):
                        with urllib.request.urlopen(url + '/view?' + urllib.parse.urlencode(image)) as response:
                            (root / (label + '.png')).write_bytes(response.read())
                print('acceptance', label, time.monotonic() - start, pid, flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
