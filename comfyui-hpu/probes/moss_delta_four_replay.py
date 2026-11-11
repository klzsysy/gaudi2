"""Replay the captured failing GDN shape with neutral tree padding."""
import json
import os
from pathlib import Path
import time
import torch
import torch.nn.functional as F
import comfy_kitchen
from comfy.text_encoders import hpu_gated_delta
import habana_frameworks.torch.utils.debug as debug

root=Path(os.environ['MOSS_NEXT_ROOT'])
mode=os.environ['MOSS_FOUR_MODE']
original_tree=hpu_gated_delta._tree_output


def padded_tree(state,query,key,value,decay,beta,program):
    rows=query.shape[-1]
    padding=8-rows
    query,key,value=(F.pad(tensor,(0,padding)) for tensor in (query,key,value))
    decay=F.pad(decay,(0,0,0,padding),value=1)
    beta=F.pad(beta,(0,0,0,padding))
    program=torch.cat((program,torch.arange(rows,8,device=program.device,dtype=program.dtype)))
    return original_tree(state,query,key,value,decay,beta,program)[:,:rows]


torch.set_grad_enabled(False)
torch.set_num_threads(16)
values=torch.load(root/'mtp-four-fences/delta-four.pt',weights_only=True)
values=[v.to('hpu').clone() if isinstance(v,torch.Tensor) else v for v in values]
if os.environ.get('MOSS_FOUR_PARAMETERS')=='1':
    for index in (1,2,10):
        values[index]=torch.nn.Parameter(values[index],requires_grad=False)
if mode=='baseline':
    function=comfy_kitchen.gated_delta_decode_deferred
elif mode=='padded':
    hpu_gated_delta._tree_output=padded_tree
    function=hpu_gated_delta._decode_deferred
else:
    function=hpu_gated_delta._decode_deferred
compiled=hpu_gated_delta.compiled_ops()[1] if mode=='product' else torch.compile(function,backend='hpu_backend',fullgraph=True,dynamic=False)
start=time.perf_counter()
output=compiled(*values).cpu()
record=dict(cold=time.perf_counter()-start,shape=list(values[0].shape),fallback=debug._get_fallback_op_count())
label=mode+('-parameters' if os.environ.get('MOSS_FOUR_PARAMETERS')=='1' else '')
saved=dict(output=output,state=values[5].cpu(),gates=values[13].cpu(),sumsq=values[14].cpu())
torch.save(saved,root/('four-'+label+'.pt'))
if mode!='baseline':
    baseline=torch.load(root/'four-baseline.pt',weights_only=True)
    record['errors']={}
    for key,actual in saved.items():
        reference=baseline[key].float()
        difference=actual.float()-reference
        record['errors'][key]=dict(rrmse=float(difference.square().mean().sqrt()/reference.square().mean().sqrt().clamp(min=1e-30)),
                                   max_abs=float(difference.abs().max()),finite=bool(actual.isfinite().all()))
(root/('four-'+label+'.json')).write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
