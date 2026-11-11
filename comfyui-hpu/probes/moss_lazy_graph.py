"""Isolated lazy/HPUGraph capability and per-request capture-cost experiment."""
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
root = Path(os.environ['MOSS_COMPLETION_ROOT'])
block = moss.Block(3072,8192,24,fused_qkv=True,dtype=torch.bfloat16,operations=comfy.ops.manual_cast)
block.load_state_dict(torch.load(root/'captures/weights-0.pt',weights_only=True))
block.to('hpu')
rows = []
for condition in (307,23):
    try:
        data=torch.load(root/f'captures/input-{condition}-0-0.pt',weights_only=True)
        source=data['args'][0].to('hpu');x=source.clone()
        mod=tuple(t.to('hpu') for t in data['args'][1]);pe=data['args'][2].to('hpu')
        options={'_moss_hpu_blockwise_rope':True}
        def run():
            x.copy_(source)
            result=block(x,mod,pe,options)
            htcore.mark_step()
            return result
        start=time.perf_counter();expected=run().cpu();cold=time.perf_counter()-start
        for _ in range(3):run()
        torch.hpu.synchronize();times=[]
        for _ in range(5):
            start=time.perf_counter()
            for _ in range(10):run()
            torch.hpu.synchronize();times.append((time.perf_counter()-start)/10)
        graph=torch.hpu.HPUGraph()
        x.copy_(source);htcore.mark_step();torch.hpu.synchronize()
        start=time.perf_counter()
        graph.capture_begin()
        out=block(x,mod,pe,options)
        graph.capture_end()
        torch.hpu.synchronize();capture=time.perf_counter()-start
        samples=[]
        for _ in range(5):
            start=time.perf_counter()
            for _ in range(10):
                x.copy_(source);htcore.mark_step();graph.replay(asynchronous=True)
            torch.hpu.synchronize();samples.append((time.perf_counter()-start)/10)
        actual=out.cpu().float();reference=expected.float()
        changed=torch.load(root/f'captures/input-{condition}-12-0.pt',weights_only=True)
        for destination,value in zip(mod,changed['args'][1]):destination.copy_(value.to('hpu'))
        x.copy_(source);htcore.mark_step();graph.replay();torch.hpu.synchronize();changed_output=out.cpu().float()
        changed_reference=run().cpu().float()
        row=dict(condition=condition,cold_s=cold,lazy_ms=1000*statistics.median(times),capture_s=capture,
                 graph_ms=1000*statistics.median(samples),lazy_samples=times,graph_samples=samples,
                 replay_error=float((actual-reference).square().mean().sqrt()/reference.square().mean().sqrt()),
                 changed_mod_error=float((changed_output-changed_reference).square().mean().sqrt()/changed_reference.square().mean().sqrt()),
                 fallback=debug._get_fallback_op_count(),memory=torch.hpu.memory_stats())
        torch.save(actual,root/f'lazy-output-{condition}.pt')
        del graph
        gc.collect()
    except Exception:
        row=dict(condition=condition,error=traceback.format_exc())
    rows.append(row);(root/'lazy-graph.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
