"""Ordinary 8189 acceptance for the saved 2MP workflow and model reload."""
import copy
import json
from pathlib import Path
import time
import urllib.request

base = Path('/home/sonny/gaudi2/comfyui-hpu')
root = base/'performance-qwen21-2026-11-15/normal'
root.mkdir(exist_ok=True)
source = json.loads((base/'workflows/qwen21-latest-2026-11-15-api.json').read_text())
original = next(iter(json.loads((root.parent/'latest-history.json').read_text()).values()))

def request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request('http://192.168.10.21:8189'+path, data=data, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        content = response.read()
    return json.loads(content) if content else None

def submit(label, variant=False):
    graph = copy.deepcopy(source)
    graph['461']['inputs']['filename_prefix'] = 'gaudi/qwen21_20261115/'+label
    if variant:
        prompt = 'A studio photograph of one red ceramic teapot with a single curved handle and a short spout, on a plain wooden table. Soft side lighting, neutral background, no text.'
        graph['459:471']['inputs'].update(prompt=prompt, thinking=False, **{'sampling_mode.seed':2})
        graph['459:474']['inputs']['on_false'] = prompt
        graph['459:488']['inputs']['noise_seed'] = 2
        graph['13']['inputs'].update(aspect_ratio='1:1 (Square)', megapixels=.25)
    payload = dict(prompt=graph)
    if not variant:
        payload['extra_data'] = original['prompt'][3]
    pid = request('/prompt', payload)['prompt_id']
    while True:
        history = request('/history/'+pid)
        if pid in history:
            break
        time.sleep(1)
    result = history[pid]
    (root/(label+'.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    assert result['status']['status_str']=='success', result['status']
    stamps = [item[1]['timestamp'] for item in result['status']['messages'] if item[0] in ('execution_start','execution_success')]
    print(label, (stamps[-1]-stamps[0])/1000, flush=True)

submit('cold')
submit('seed2-512', variant=True)
request('/free', dict(unload_models=True, free_memory=True))
samples = []
for _ in range(9):
    time.sleep(10)
    samples.append(request('/system_stats')['devices'])
    if len(samples)>=3 and len({s[0]['torch_vram_total'] for s in samples[-3:]})==1:
        break
(root/'free.json').write_text(json.dumps(samples, indent=2))
submit('reload')
(root/'queue.json').write_text(json.dumps(request('/queue'), indent=2))
