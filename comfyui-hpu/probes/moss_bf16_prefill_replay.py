"""Fixed-input GDN prefill compilation, including recurrent-state accuracy."""
import json
import os
from pathlib import Path
import statistics
import time

import torch
from comfy.cli_args import args
args.disable_dynamic_vram=True
args.disable_pinned_memory=True
args.disable_async_offload=True
args.use_pytorch_cross_attention=True
from comfy.text_encoders.qwen35 import torch_chunk_gated_delta_rule
from comfy.text_encoders import hpu_gated_delta
import habana_frameworks.torch.utils.debug as debug

torch.set_grad_enabled(False)
torch.set_num_threads(16)
root=Path(os.environ['MOSS_BF16_ROOT'])
function=torch.compile(torch_chunk_gated_delta_rule,backend='hpu_backend',dynamic=False,options={'force_static_compile':True})
if os.environ.get('MOSS_PREFILL_BUCKETED') == '1':
    function=hpu_gated_delta.compiled_prefill(torch_chunk_gated_delta_rule)
results=[]


def hpu(v):
    if isinstance(v,torch.Tensor):return v.to('hpu')
    if isinstance(v,(tuple,list)):return type(v)(hpu(x) for x in v)
    if isinstance(v,dict):return {k:hpu(x) for k,x in v.items()}
    return v


for p in sorted((root/'prefill-capture').glob('prefill-*.pt')):
    data=torch.load(p,weights_only=True)
    values,kwargs=hpu(data['args']),hpu(data['kwargs'])
    expected=None
    for mode,fn in [('eager',torch_chunk_gated_delta_rule),('compiled',function)]:
        start=time.perf_counter();actual=fn(*values,**kwargs);torch.hpu.synchronize();cold=time.perf_counter()-start
        actual=tuple(v.cpu() if v is not None else None for v in actual)
        if expected is None:expected=actual
        errors=[]
        for a,b in zip(actual,expected):
            if a is None:errors.append(None);continue
            d=a.float()-b.float()
            errors.append({'rrmse':float(d.square().mean().sqrt()/b.float().square().mean().sqrt()),'max_abs':float(d.abs().max()),'finite':bool(a.isfinite().all())})
        for _ in range(2):fn(*values,**kwargs)
        torch.hpu.synchronize();samples=[]
        for _ in range(5):
            start=time.perf_counter()
            for _ in range(3):fn(*values,**kwargs)
            torch.hpu.synchronize();samples.append((time.perf_counter()-start)/3)
        row={'capture':p.name,'mode':mode,'cold_s':cold,'median_ms':1000*statistics.median(samples),'samples_ms':[1000*v for v in samples],'errors':errors,'fallback':debug._get_fallback_op_count()}
        row['bucketed'] = os.environ.get('MOSS_PREFILL_BUCKETED') == '1'
        row['input_dtype'] = str(values[0].dtype)
        row['input_shape'] = list(values[0].shape)
        results.append(row);(root/os.environ.get('MOSS_PREFILL_RESULT', 'prefill-replay.json')).write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
