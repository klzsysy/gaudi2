"""42 real layers: request-owned lazy graphs, including all preparation costs."""
import gc
import json
import os
from pathlib import Path
import statistics
import time
import traceback
import torch
import habana_frameworks.torch.core as htcore
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True
args.disable_comfy_compiler = True
import comfy.ops
from comfy.ldm.moss_image import model as moss
import habana_frameworks.torch.utils.debug as debug

torch.set_grad_enabled(False)
torch.set_num_threads(16)
root=Path(os.environ['MOSS_COMPLETION_ROOT'])
records=[]
blocks=[]
for layer in range(42):
    b=moss.Block(3072,8192,24,fused_qkv=True,dtype=torch.bfloat16,operations=comfy.ops.manual_cast)
    b.load_state_dict(torch.load(root/f'captures/weights-{layer}.pt',weights_only=True))
    blocks.append(b.to('hpu'))
for condition in (307,23):
    try:
        data=torch.load(root/f'captures/input-{condition}-0-0.pt',weights_only=True)
        source=data['args'][0].to('hpu');mod=tuple(v.to('hpu') for v in data['args'][1]);pe=data['args'][2].to('hpu')
        def chain(x):
            for block in blocks:x=block(x,mod,pe,{'_moss_hpu_blockwise_rope':True})
            return x
        requests=[]
        for request in range(3):
            start=time.perf_counter()
            x=source.clone();htcore.mark_step();torch.hpu.synchronize()
            allocation=time.perf_counter()-start
            begin=time.perf_counter();chain(x);htcore.mark_step();torch.hpu.synchronize();warmup=time.perf_counter()-begin
            x.copy_(source);htcore.mark_step();torch.hpu.synchronize()
            begin=time.perf_counter();graph=torch.hpu.HPUGraph();graph.capture_begin();out=chain(x);graph.capture_end();torch.hpu.synchronize();capture=time.perf_counter()-begin
            times=[]
            for _ in range(5):
                begin=time.perf_counter()
                x.copy_(source);htcore.mark_step();graph.replay(asynchronous=True);torch.hpu.synchronize()
                times.append(time.perf_counter()-begin)
            actual=out.cpu().float();ref=torch.load(root/f'rotating-output-{condition}.pt',weights_only=True)
            row=dict(condition=condition,request=request,allocation_s=allocation,warmup_s=warmup,capture_s=capture,
                     replay_s=statistics.median(times),samples_s=times,eager_rrmse=float((actual-ref).square().mean().sqrt()/ref.square().mean().sqrt()),
                     estimated_25_forward_s=allocation+warmup+capture+25*statistics.median(times),fallback=debug._get_fallback_op_count(),memory=torch.hpu.memory_stats())
            records.append(row);(root/'lazy-chain.json').write_text(json.dumps(records,indent=2));print(json.dumps(row),flush=True)
            del graph,out,x
            gc.collect();htcore.mark_step();torch.hpu.synchronize()
    except Exception:
        records.append(dict(condition=condition,error=traceback.format_exc()))
        (root/'lazy-chain.json').write_text(json.dumps(records,indent=2));print(records[-1],flush=True)
try:
    function=torch.compile(lambda x:x.sin()+x,backend='hpu_backend')
    x=torch.ones(16,device='hpu');out=function(x);htcore.mark_step();torch.hpu.synchronize()
    result=dict(compile_in_lazy='accepted',finite=bool(out.cpu().isfinite().all()))
except Exception:
    result=dict(compile_in_lazy='failed',error=traceback.format_exc())
(root/'lazy-compile-compatibility.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
