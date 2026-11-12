"""Check native SiTU against real inputs and compiled production math."""
import json
import os
from pathlib import Path
import statistics
import time

import torch
import habana_frameworks.torch
import habana_frameworks.torch.utils.debug as debug


root=Path(os.environ['MOSS_OPT_ROOT'])
torch.ops.load_library(str(root/'native-build/moss_situ_bridge.so'))
torch.set_grad_enabled(False)
torch.set_num_threads(32)


@torch.library.register_fake('custom_op::moss_situ')
def situ_meta(x):
    return x.new_empty((x.shape[0],x.shape[1]//2))


@torch.library.register_fake('custom_op::moss_situ_quant')
def quant_meta(x):
    return x.new_empty((x.shape[0],x.shape[1]//2),dtype=torch.float8_e4m3fn),x.new_empty((x.shape[0],1),dtype=torch.float32)


def baseline(x):
    hidden,gate=x.chunk(2,dim=-1)
    return (4*torch.tanh(gate/4)*torch.sigmoid(gate))*(25*torch.tanh(hidden/25))


def baseline_quant(x):
    output=baseline(x)
    scale=output.abs().amax(-1,keepdim=True).float().clamp_min(1e-12)/240
    return torch.ops.hpu.cast_to_fp8_v2(output,scale.reciprocal(),False,False,torch.float8_e4m3fn)[0],scale


functions={'baseline':baseline,'native':torch.ops.custom_op.moss_situ.default,
           'baseline_quant':baseline_quant,'native_quant':torch.ops.custom_op.moss_situ_quant.default}
compiled={name:torch.compile(function,backend='hpu_backend',dynamic=False) for name,function in functions.items()}


def measure(function,x):
    for _ in range(3):output=function(x)
    torch.hpu.synchronize()
    samples=[]
    for _ in range(5):
        start=time.perf_counter_ns()
        for _ in range(20):output=function(x)
        torch.hpu.synchronize()
        samples.append((time.perf_counter_ns()-start)/1e9/20)
    return output,statistics.median(samples)


records=[]
for layer in [0,20,41]:
    source=torch.load(root/f'situ-input-{layer}.pt',weights_only=True).reshape(-1,16384)
    x=source.to('hpu')
    reference=baseline(source.float())
    for name,function in compiled.items():
        output,seconds=measure(function,x)
        if isinstance(output,tuple):
            value=output[0].cpu().float()*output[1].cpu()
        else:value=output.cpu().float()
        record=dict(layer=layer,mode=name,seconds=seconds,rrmse=float((value-reference).norm()/reference.norm()),
                    finite=bool(value.isfinite().all()),fallback=debug._get_fallback_op_count())
        records.append(record)
        (root/'native-screen.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(record),flush=True)
