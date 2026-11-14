"""Check shared BF16 MTP callers through the ordinary server, then restore Qwen."""
import argparse
import copy
import json
from pathlib import Path
import time
import urllib.request

base = Path('/home/sonny/gaudi2/comfyui-hpu')
root = base/'performance-qwen21-2026-11-15'
old = base/'performance-2026-11-12-three-stage/quality-product'
p = argparse.ArgumentParser()
p.add_argument('--label', default='normal')
p.add_argument('--no-restore', action='store_true')
p.add_argument('--capture-only', action='store_true')
p.add_argument('--restore-only', action='store_true')
p.add_argument('--reference-dir', type=Path)
a = p.parse_args()
output = root/a.label
output.mkdir(exist_ok=True)

def request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request('http://192.168.10.21:8189'+path, data=data, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)

def run(label, graph, extra=None):
    payload = dict(prompt=graph)
    if extra is not None:
        payload['extra_data'] = extra
    pid = request('/prompt', payload)['prompt_id']
    while pid not in (history := request('/history/'+pid)):
        time.sleep(1)
    result = history[pid]
    (output/(label+'.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    assert result['status']['status_str']=='success', result['status']
    return result

records = []
for label in (() if a.restore_only else ('teapot','poster','person')):
    reference_path = a.reference_dir/f'mtp-{label}.json' if a.reference_dir else old/f'bf16-{label}-history.json'
    reference = json.loads(reference_path.read_text())
    original = reference['prompt'][2]
    graph = {}
    def collect(node):
        if node in graph:
            return
        graph[node] = copy.deepcopy(original[node])
        for value in graph[node]['inputs'].values():
            if isinstance(value, list) and len(value)==2 and isinstance(value[0],str) and value[0] in original:
                collect(value[0])
    collect('18:13')
    result = run('mtp-'+label, graph)
    records.append(dict(case=label, text_exact=result['outputs']['18:13']==reference['outputs']['18:13'],
                        mtp=graph['18:11']['inputs']['mtp']))
    print(records[-1], flush=True)
if records:
    (root/('mtp-regression.json' if a.label=='normal' else a.label+'-regression.json')).write_text(json.dumps(records, indent=2))
if not a.capture_only:
    assert all(row['text_exact'] for row in records), records
if a.no_restore:
    raise SystemExit(0)

graph = json.loads((base/'workflows/qwen21-latest-2026-11-15-api.json').read_text())
graph['461']['inputs']['filename_prefix']='gaudi/qwen21_20261115/final'
original = next(iter(json.loads((root/'latest-history.json').read_text()).values()))
run('final', graph, original['prompt'][3])
(root/'normal/queue.json').write_text(json.dumps(request('/queue'), indent=2))
