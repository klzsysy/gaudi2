"""Ordinary-server acceptance, stable unload, and full BF16 reload."""
import argparse
import copy
import json
from pathlib import Path
import time
import urllib.request

p=argparse.ArgumentParser()
p.add_argument('output',type=Path)
p.add_argument('--url',default='http://192.168.10.21:8189')
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
base=Path(__file__).resolve().parents[1]
source=json.loads((base/'workflows/qwen21-user-bf16-api.json').read_text())
ui=json.loads((base/'workflows/qwen21-user-bf16.json').read_text())

def request(path,payload=None):
    req=urllib.request.Request(a.url+path,data=json.dumps(payload).encode() if payload is not None else None,
                               headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=30) as response:content=response.read()
    return json.loads(content) if content else None


def submit(label,seed):
    graph=copy.deepcopy(source)
    workflow=copy.deepcopy(ui)
    graph['459:458']['inputs']['seed']=seed
    graph['459:471']['inputs']['sampling_mode.seed']=seed
    next(node for node in workflow['nodes'] if node['id']==459)['widgets_values'][11]=seed
    graph['461']['inputs']['filename_prefix']='gaudi/qwen21_bf16/normal_'+label
    pid=request('/prompt',dict(prompt=graph,extra_data={'extra_pnginfo':{'workflow':workflow}}))['prompt_id']
    print(label,pid,flush=True)
    while True:
        history=request('/history/'+pid)
        if pid in history:break
        time.sleep(1)
    result=history[pid]
    (a.output/(label+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    assert result['status']['status_str']=='success',result['status']
    times=[message[1]['timestamp'] for message in result['status']['messages'] if message[0] in ('execution_start','execution_success')]
    print(label,(times[-1]-times[0])/1000,flush=True)

submit('cold',1)
submit('new-seed',2)
request('/free',dict(unload_models=True,free_memory=True))
samples=[]
for _ in range(9):
    time.sleep(10)
    stats=request('/system_stats')['devices']
    samples.append(stats)
    print('unload',stats[0]['torch_vram_total'],flush=True)
    if len(samples)>=3 and len({s[0]['torch_vram_total'] for s in samples[-3:]})==1:break
(a.output/'free.json').write_text(json.dumps(samples,indent=2))
submit('reload',1)
(a.output/'queue.json').write_text(json.dumps(request('/queue'),indent=2))
