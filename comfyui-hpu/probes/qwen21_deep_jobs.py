"""Run controlled full/cached Qwen BF16 workflows in a dedicated server process."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

parser=argparse.ArgumentParser()
parser.add_argument('label')
parser.add_argument('--trial',default='')
parser.add_argument('--runs',type=int,default=6)
parser.add_argument('--cached',type=int,default=5)
parser.add_argument('--diagnostics',action='store_true')
parser.add_argument('--capture',action='store_true')
parser.add_argument('--stages',action='store_true')
parser.add_argument('--quality-cases',action='store_true')
parser.add_argument('--free',action='store_true')
parser.add_argument('--workflow',type=Path)
parser.add_argument('--output-root',type=Path)
parser.add_argument('--case-graphs',type=Path)
parser.add_argument('--teacher',type=Path)
parser.add_argument('--reload',action='store_true')
parser.add_argument('--server-module', default='qwen21_deep_server', choices=('qwen21_deep_server', 'qwen21_fp8_server'))
parser.add_argument('--fp8-diagnostics', action='store_true')
parser.add_argument('--native-dit', action='store_true')
parser.add_argument('--native-clip', action='store_true')
a=parser.parse_args()
base=Path('/home/sonny/gaudi2/comfyui-hpu')
root=(a.output_root or base/'performance-qwen21-bf16-2026-11-13')/a.label
root.mkdir(parents=True,exist_ok=True)
workflow=json.loads((a.workflow or base/'workflows/qwen21-user-bf16-api.json').read_text())
teacher=json.loads(a.teacher.read_text()) if a.teacher else None
env=dict(os.environ,PT_HPU_LAZY_MODE='0',GC_KERNEL_PATH=os.environ.get('GC_KERNEL_PATH','/usr/lib/habanalabs/libtpc_kernels.so'),
         PYTHONPATH=os.pathsep.join((str(Path.cwd()),str(Path(__file__).parent))),QWEN_DEEP_DIR=str(root),MOSS_EVENTS=str(root/'events.jsonl'),QWEN21_TRIAL=a.trial)
url='http://127.0.0.1:8190'
log=(root/'server.log').open('w')
server=subprocess.Popen([sys.executable,'-c',f'import {a.server_module}; {a.server_module}.main()',
    '--disable-dynamic-vram','--disable-pinned-memory','--disable-async-offload','--disable-xformers',
    '--use-pytorch-cross-attention','--disable-all-custom-nodes','--listen','127.0.0.1','--port','8190'],env=env,stdout=log,stderr=subprocess.STDOUT)

def request(path,payload=None):
    req=urllib.request.Request(url+path,data=json.dumps(payload).encode() if payload is not None else None,
                               headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=30) as response:
            content=response.read()
        return json.loads(content) if content else None
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode()) from e

try:
    for _ in range(90):
        if server.poll() is not None:raise RuntimeError('server exited during startup')
        try:
            request('/queue');break
        except urllib.error.URLError:time.sleep(1)
    else:raise RuntimeError('server unavailable')
    sequence=[('full',i,'') for i in range(a.runs)]+[('cached',i,'') for i in range(a.cached)]
    cases=json.loads(a.case_graphs.read_text()) if a.case_graphs else {}
    sequence += [(label,0,'') for label in cases]
    if a.reload:
        sequence += [('reload',0,'')]
    if a.quality_cases:
        sequence += [(label,0,'') for label in ('seed2','size512','size1024','thinking','official')]
    if a.stages:
        sequence += [('stages',0,'stages')]
    if a.capture:
        sequence += [('capture',0,'capture')]
    if a.diagnostics:
        sequence += [('stages',0,'stages'),('window_text',0,'window_text'),('trace_text',0,'trace_text'),('trace_dit',0,'trace_dit')]
    if a.fp8_diagnostics:
        sequence += [('window_text',0,'window_text'),('trace_text',0,'trace_text'),('trace_dit',0,'trace_dit')]
    if a.native_dit:
        sequence += [('window_dit',0,'window_dit'),('native_dit',0,'native_dit')]
    if a.native_clip:
        sequence += [('window_clip',0,'window_clip'),('native_clip',0,'native_clip')]
    for label,index,profile in sequence:
        if label=='reload':
            request('/free',dict(unload_models=True,free_memory=True))
            samples=[]
            for _ in range(4):
                time.sleep(10)
                samples.append(request('/system_stats')['devices'])
            (root/'reload-free.json').write_text(json.dumps(samples,indent=2))
        graph=copy.deepcopy(cases.get(label,workflow))
        if label=='seed2':
            graph['459:458']['inputs']['seed']=2
            graph['459:471']['inputs']['sampling_mode.seed']=2
        if label in ('size512','size1024'):
            graph['13']['inputs'].update(aspect_ratio='1:1 (Square)',megapixels=.25 if label=='size512' else 1.)
        if label=='thinking':graph['459:471']['inputs']['thinking']=True
        if label=='official':
            sampler=graph['459:458']['inputs']
            graph['900']={'class_type':'RandomNoise','inputs':{'noise_seed':sampler['seed']}}
            graph['901']={'class_type':'CFGGuider','inputs':{k:sampler[k] for k in ('model','positive','negative','cfg')}}
            graph['902']={'class_type':'KSamplerSelect','inputs':{'sampler_name':'euler'}}
            graph['903']={'class_type':'ManualSigmas','inputs':{'sigmas':'1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0'}}
            graph['459:458']={'class_type':'SamplerCustomAdvanced','inputs':dict(noise=['900',0],guider=['901',0],sampler=['902',0],sigmas=['903',0],latent_image=sampler['latent_image'])}
        graph['461']['inputs']['filename_prefix']=f'gaudi/qwen21_bf16/{a.label}_{label}_{index}'
        nodes=['KSampler','SamplerCustomAdvanced','VAEDecode']
        if label!='cached' and profile not in ('trace_dit','capture'):nodes+=['TextGenerate','TextEncodeQwenImage21']
        pid=request('/prompt',dict(prompt=graph,extra_data=dict(moss_recompute_nodes=nodes,qwen_profile=profile,
                    qwen_teacher=teacher if label=='full' else None)))['prompt_id']
        print(a.label,label,index,pid,flush=True)
        while True:
            if server.poll() is not None:raise RuntimeError('server exited during execution')
            try:
                history=request('/history/'+pid)
            except TimeoutError:
                continue
            if pid in history:break
            time.sleep(.5)
        result=history[pid]
        (root/f'{label}-{index}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        assert result['status']['status_str']=='success',result['status']
        stamps=[m[1]['timestamp'] for m in result['status']['messages'] if m[0] in ('execution_start','execution_success')]
        print('seconds',(stamps[-1]-stamps[0])/1000,flush=True)
    if a.free:
        request('/free',dict(unload_models=True,free_memory=True))
        samples=[]
        for delay in (5,10,10,10,10,10):
            time.sleep(delay)
            samples.append(request('/system_stats')['devices'])
        (root/'free.json').write_text(json.dumps(samples,indent=2))
finally:
    server.terminate()
    try:server.wait(timeout=40)
    except subprocess.TimeoutExpired:server.kill();server.wait()
    log.close()
