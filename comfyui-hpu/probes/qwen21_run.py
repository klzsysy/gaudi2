"""Measure real API executions, retaining loader outputs in the observer."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

p=argparse.ArgumentParser()
p.add_argument('workflow',type=Path)
p.add_argument('output',type=Path)
p.add_argument('--runs',type=int,default=1)
p.add_argument('--no-enhancer',action='store_true')
p.add_argument('--cached-text',action='store_true')
p.add_argument('--normal-server',action='store_true')
p.add_argument('--megapixels',type=float)
p.add_argument('--prompt')
p.add_argument('--url',default='http://192.168.10.21:8189')
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
workflow=json.loads(a.workflow.read_text())
if a.no_enhancer:
    for key in ['471','473','475']:
        workflow.pop(key,None)
    workflow['474']['inputs'].pop('on_true',None)
if a.megapixels:
    workflow['13']['inputs']['megapixels']=a.megapixels
if a.prompt:
    workflow['474']['inputs']['on_false']=a.prompt
    if '471' in workflow:
        workflow['471']['inputs']['prompt']=a.prompt
for run in range(a.runs):
    if a.normal_server:
        workflow['488']['inputs']['noise_seed'] = 43 + run
        if '471' in workflow:
            workflow['471']['inputs']['sampling_mode.seed'] = 43 + run
    workflow['461']['inputs']['filename_prefix']='gaudi/qwen21_'+a.output.name+'_'+str(run)
    extra={'moss_recompute_nodes':['SamplerCustomAdvanced','VAEDecode']}
    if not a.cached_text:
        extra['moss_recompute_nodes']+=['TextGenerate','TextEncodeQwenImage21']
    request=urllib.request.Request(a.url+'/prompt',data=json.dumps({'prompt':workflow,'extra_data':{} if a.normal_server else extra}).encode(),headers={'Content-Type':'application/json'})
    try:
        submitted=json.load(urllib.request.urlopen(request))
    except urllib.error.HTTPError as e:
        print(e.read().decode(),flush=True)
        raise
    pid=submitted['prompt_id']
    print(a.output.name,run,pid,flush=True)
    started=time.monotonic()
    while True:
        history=json.load(urllib.request.urlopen(a.url+'/history/'+pid))
        if pid in history:
            result=history[pid]
            result['client_seconds']=time.monotonic()-started
            (a.output/f'{run}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
            stamps=[m[1]['timestamp'] for m in result['status']['messages'] if m[0] in ['execution_start','execution_success','execution_error']]
            print(json.dumps({'run':run,'status':result['status']['status_str'],'seconds':(stamps[-1]-stamps[0])/1000,'images':result['outputs'].get('461',{}).get('images',[])}),flush=True)
            if result['status']['status_str']!='success':
                print(result['status'],flush=True)
                raise SystemExit(1)
            break
        time.sleep(1)
