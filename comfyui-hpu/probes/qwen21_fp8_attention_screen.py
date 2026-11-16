"""Benchmark existing HPU attention kernels on captured Qwen21 tensors."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch
from comfy.cli_args import args
args.use_pytorch_cross_attention = True
from comfy.ldm.qwen_image21 import model as dit
from comfy.ldm.modules.attention import _hpu_fp8_attention_av, AttentionTensorContainer
from habana_frameworks.torch.hpex.kernels import FusedSDPA

parser = argparse.ArgumentParser()
parser.add_argument('capture', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--quick', action='store_true')
a = parser.parse_args()
torch.set_num_threads(32)
torch.set_grad_enabled(False)
saved = torch.load(a.capture, weights_only=False)
q, k, v = [saved[name].to('hpu') for name in ('q', 'k', 'v')]
heads = saved['heads']


def ordinary(q,k,v):
    return dit._attention(q,k,v,heads,None,{},None)


def fused(q,k,v,group,mode,recompute):
    q,k,v = [x.view(x.shape[0], x.shape[1], heads, -1).transpose(1,2) for x in (q,k,v)]
    outputs = [FusedSDPA.apply(q[:,h:h+group],k[:,h:h+group],v[:,h:h+group],None,0.0,False,None,mode,recompute)
               for h in range(0,heads,group)]
    return torch.cat(outputs,dim=1).transpose(1,2).reshape(q.shape[0],q.shape[2],-1)


def mixed(q,k,v):
    q,k,v = [x.view(x.shape[0], x.shape[1], heads, -1).transpose(1,2) for x in (q,k,v)]
    outputs = [_hpu_fp8_attention_av(q[:,h:h+1],k[:,h:h+1],v[:,h:h+1]) for h in range(heads)]
    return torch.cat(outputs,dim=1).transpose(1,2).reshape(q.shape[0],q.shape[2],-1)


reference = ordinary(q,k,v).cpu().float()
rows_to_check = [0,127,511,4095,q.shape[1]-1]
heads_to_check = [0,15,31]
cpu_q,cpu_k,cpu_v = [saved[name].reshape(1,-1,heads,128)[0] for name in ('q','k','v')]
accurate = []
for head in heads_to_check:
    scores = cpu_q[rows_to_check,head].double() @ cpu_k[:,head].double().T / 128**.5
    accurate.append(scores.softmax(-1) @ cpu_v[:,head].double())
accurate = torch.stack(accurate)

rows = []
candidates = [('ordinary',ordinary),('mixed',mixed)]
for group in (32,8,1):
    for mode in ('None','fast','fp32'):
        for recompute in (False,True):
            candidates.append((f'{mode}-{group}-recompute{recompute}',
                lambda q,k,v,group=group,mode=mode,recompute=recompute:fused(q,k,v,group,mode,recompute)))
if a.quick:
    candidates = [(n,f) for n,f in candidates if n in ('ordinary','mixed','fast-32-recomputeFalse','fp32-32-recomputeFalse')]
for name,function in candidates:
    gc.collect()
    torch._dynamo.reset()
    compiled = torch.compile(function,backend='hpu_backend',dynamic=False,options={'force_static_compile':True})
    result = compiled(q,k,v)
    torch.hpu.synchronize()
    cpu = result.cpu().float()
    error = float((cpu-reference).norm()/reference.norm())
    times=[]
    for _ in range(5):
        torch.hpu.synchronize()
        start=time.perf_counter()
        for repeat in range(4):result=compiled(q,k,v)
        torch.hpu.synchronize()
        times.append((time.perf_counter()-start)/4)
    sampled = torch.stack([cpu.reshape(1,-1,heads,128)[0,rows_to_check,head].double() for head in heads_to_check])
    row=dict(name=name,seconds=statistics.median(times),rrmse=error,fp64_sample_rrmse=float((sampled-accurate).norm()/accurate.norm()),finite=bool(cpu.isfinite().all()))
    rows.append(row)
    print(json.dumps(row),flush=True)
    a.output.write_text(json.dumps(dict(shape={k:list(saved[k].shape) for k in ('q','k','v')},results=rows),indent=2))
