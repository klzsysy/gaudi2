"""Query tiling of existing BF16 SDPA; normalization and math stay unchanged."""
import json
import os
from pathlib import Path
import statistics
import time

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True
import comfy.ops
from comfy.ldm.moss_image import model as moss
import moss_bf16_candidates as candidates
from habana_frameworks.torch.hpex.kernels import FusedSDPA
import habana_frameworks.torch.utils.debug as debug

torch.set_grad_enabled(False)
torch.set_num_threads(16)
root = Path(os.environ['MOSS_BF16_ROOT'])
candidates.install('adaln-rotary')
state = torch.load(root / 'captures/weights-0.pt', weights_only=True)
block = moss.Block(3072, 8192, 24, fused_qkv=True, dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
block.load_state_dict(state)
block.to('hpu')
original = comfy.ops.scaled_dot_product_attention
records = []


def measure(fn):
    value = fn().cpu().double()
    for _ in range(3):fn()
    torch.hpu.synchronize()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(10):fn()
        torch.hpu.synchronize()
        samples.append((time.perf_counter()-start)/10)
    return value, {'median_ms': statistics.median(samples)*1000, 'samples_ms': [v*1000 for v in samples]}


def error(value, reference):
    return float((value-reference).square().mean().sqrt()/reference.square().mean().sqrt())


for condition in (307,23):
    data = torch.load(root / f'captures/input-{condition}-0-0.pt', weights_only=True)
    x = data['args'][0].to('hpu')
    mod = tuple(t.to('hpu') for t in data['args'][1])
    pe = data['args'][2].to('hpu')
    reference64 = torch.load(root / f'fp64-{condition}-0-0.pt', weights_only=True)
    captured = []
    def capture(q, k, v, *values, **kwargs):
        captured.append((q,k,v))
        return original(q,k,v,*values,**kwargs)
    comfy.ops.scaled_dot_product_attention = capture
    block._forward_attention(moss.modulate(x,block.norm1,mod,0,None), pe,{})
    comfy.ops.scaled_dot_product_attention = original
    q,k,v = captured[0]
    for mode in os.environ.get('MOSS_SDPA_CASES','math,math-256,math-512,math-1024,native,native-256,native-512').split(','):
        chunk = int(mode.split('-')[1]) if '-' in mode else q.shape[-2]
        def attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
            assert attn_mask is None and dropout_p == 0 and not is_causal
            pieces = []
            for start in range(0,q.shape[-2],chunk):
                part = q[...,start:start+chunk,:]
                if mode.startswith('native'):
                    out = FusedSDPA.apply(part,k,v,None,0.,False,None,'None',False)
                else:
                    out = original(part,k,v,dropout_p=0.,is_causal=False)
                pieces.append(out)
            return torch.cat(pieces,dim=-2) if len(pieces)>1 else pieces[0]
        torch._dynamo.reset()
        moss._compiled_block_ops.cache_clear()
        comfy.ops.scaled_dot_product_attention = attention
        compiled = torch.compile(attention,backend='hpu_backend',dynamic=False,options={'force_static_compile':True})
        av,at=measure(lambda:compiled(q,k,v))
        bv,bt=measure(lambda:block(x.clone(),mod,pe,{}))
        if mode=='math':baseline_a,baseline_b=av,bv
        record={'condition':condition,'mode':mode,'attention':at,'block':bt,
                'attention_rrmse':error(av,baseline_a),'block_rrmse':error(bv,baseline_b),
                'fp64_rrmse':error(bv,reference64),'fallback':debug._get_fallback_op_count()}
        records.append(record)
        (root/'attention-tiling.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(record),flush=True)
