"""Natural text/image pairs, MTP modes and INT8 compatibility for deferred GDN."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.parse
import urllib.request

root = Path(sys.argv[1])
destination = root/'quality'
destination.mkdir(exist_ok=True)
source = json.loads((root/'source.json').read_text())
url = 'http://127.0.0.1:8190'


def api(route,data=None):
    request = urllib.request.Request(url+route,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=30) as response:
        body = response.read()
        return json.loads(body) if body else None


cases = [
    ('teapot',.25,'A red ceramic teapot on a wooden table beside a sunlit window, product photography.'),
    ('poster',.5625,'A clean poster featuring a blue ceramic vase, three yellow flowers, a white background, and large readable text "MOSS" at the top.'),
    ('person',1.,'Portrait photograph of an elderly woman wearing a red scarf, detailed face and hands, holding a small white teacup, natural window light.'),
]
records = json.loads((destination/'results.json').read_text()) if (destination/'results.json').exists() else []
selected = os.environ.get('MOSS_QUALITY_LABELS')
selected = set(selected.split(',')) if selected else None


def run(label,graph,baseline=False):
    if selected is not None and label not in selected:
        return
    started = time.perf_counter()
    pid = api('/prompt',dict(prompt=graph,extra_data=dict(moss_recompute=True,moss_diagnostic=True,moss_tree_baseline=baseline)))['prompt_id']
    while pid not in (history:=api('/history/'+pid)):
        if time.perf_counter()-started>1200:
            raise TimeoutError(pid)
        time.sleep(.2)
    history = history[pid]
    (destination/(label+'-history.json')).write_text(json.dumps(history))
    assert history['status']['status_str']=='success',history['status']
    events = [json.loads(s) for s in (root/'events.jsonl').read_text().splitlines()]
    events = [e for e in events if e.get('prompt_id')==pid]
    generated = next(e for e in events if e['kind']=='generate_end')
    assert generated['output_tokens'] < graph['18:11']['inputs']['max_length']
    checks = [e for e in events if e['kind']=='tensor_check']
    assert len(checks)>=2 and all(e['finite'] for e in checks)
    assert not next(e for e in events if e['kind']=='request_end')['fallback_counts']
    for output in history['outputs'].values():
        for image in output.get('images',[]):
            with urllib.request.urlopen(url+'/view?'+urllib.parse.urlencode(image)) as response:
                (destination/(label+'.png')).write_bytes(response.read())
    records[:] = [row for row in records if row['label'] != label]
    records.append(dict(label=label,prompt_id=pid,tokens=generated['output_tokens'],text=generated['text'],checks=checks))
    (destination/'results.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))
    print(label,generated['output_tokens'],'PASS',flush=True)


for name,size,prompt in cases:
    if selected is not None and not {name+'-baseline',name+'-candidate'} & selected:
        continue
    if name=='person':
        released = destination/'reload'
        released.mkdir(exist_ok=True)
        subprocess.run([sys.executable,str(Path(__file__).with_name('moss_memory_release.py')),str(released),url],check=True)
    for baseline in (True,False):
        graph = copy.deepcopy(source)
        graph['18:17']['inputs']['value'] = prompt
        graph['19']['inputs']['megapixels'] = size
        graph['18:7']['inputs']['seed'] = 42
        run(name+('-baseline' if baseline else '-candidate'),graph,baseline)

for mode in ('off','3','tree3s1'):
    graph = copy.deepcopy(source)
    graph['18:17']['inputs']['value'] = cases[0][2]
    graph['19']['inputs']['megapixels'] = .25
    graph['18:11']['inputs']['mtp'] = mode
    run('mtp-'+mode,graph)

graph = copy.deepcopy(source)
graph['18:2']['inputs']['clip_name'] = 'qwen3.5_9b_int8_convrot.safetensors'
graph['19']['inputs']['megapixels'] = .25
run('int8',graph)
