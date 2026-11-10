"""Check dynamic DiT compilation and reuse Kitchen tensor implementations."""
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True
import comfy.ops
import comfy.ldm.moss_image.model as moss
if hasattr(moss, '_attention'):
    moss.attention = moss._attention
from comfy_kitchen.backends.eager.adaln import rms_adaln
from comfy_kitchen.backends.eager.rope import rms_rope_split_half
import habana_frameworks.torch.utils.debug as debug

torch.set_grad_enabled(False)
torch.set_num_threads(16)
root = Path(os.environ['MOSS_DIT_ROOT'])
source = torch.load(root / 'original/block-input-0.pt', weights_only=True)
state = source['state']
block = moss.Block(state['norm1.weight'].numel(), state['proj.ffn_up_gate.weight'].shape[0] // 2, source['heads'], fused_qkv=True, dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
block.load_state_dict(state)
block.to('hpu')
x = source['args'][0].to('hpu')
mod = tuple(t.to('hpu') for t in source['args'][1])
pe = source['args'][2].to('hpu')
def fp16_linear_wanted(x):
    return x.is_cuda and x.dtype == torch.float16 and getattr(torch.backends.cuda.matmul, 'allow_fp16_accumulation', False)
comfy.ops._fp16_linear_wanted = fp16_linear_wanted
original_attention = moss.attention
original_adaln = moss.comfy.quant_ops.ck.rms_adaln
original_rope = moss.comfy.quant_ops.ck.rms_rope_split_half
original_sdpa = comfy.ops.scaled_dot_product_attention
original_modulation = moss.modulated_rms_norm
original_situ = comfy.ops.INPUT_ACT_EAGER['situ']
results = {}

for mode in os.environ.get('MOSS_SHAPE_MODES', 'attention-dynamic,block-dynamic,block-inline').split(','):
    torch._dynamo.reset()
    moss.attention = original_attention
    moss.comfy.quant_ops.ck.rms_adaln = original_adaln
    moss.comfy.quant_ops.ck.rms_rope_split_half = original_rope
    comfy.ops.scaled_dot_product_attention = original_sdpa
    moss.modulated_rms_norm = original_modulation
    comfy.ops.INPUT_ACT_EAGER['situ'] = original_situ
    if mode == 'attention-dynamic':
        moss.attention = torch.compile(original_attention, backend='hpu_backend', fullgraph=True, dynamic=True)
        candidate = block
    elif mode.startswith('sdpa-'):
        def static_head(q, k, v, *values, **kwargs):
            for tensor in (q, k, v):
                torch._dynamo.mark_static(tensor, 1)
                torch._dynamo.mark_static(tensor, 3)
            return original_sdpa(q, k, v, *values, **kwargs)
        comfy.ops.scaled_dot_product_attention = torch.compile(static_head if mode == 'sdpa-static-head' else original_sdpa, backend='hpu_backend', fullgraph=True, dynamic=None if mode == 'sdpa-auto' else True)
        candidate = block
    else:
        if mode == 'block-inline':
            moss.comfy.quant_ops.ck.rms_adaln = rms_adaln
            moss.comfy.quant_ops.ck.rms_rope_split_half = rms_rope_split_half
        if mode in ('block-no-mod', 'block-rounding'):
            moss.modulated_rms_norm = torch.compiler.disable(original_modulation)
        if mode in ('block-no-situ', 'block-rounding'):
            comfy.ops.INPUT_ACT_EAGER['situ'] = torch.compiler.disable(original_situ)
        candidate = torch.compile(block, backend='hpu_backend', fullgraph=mode not in ('block-no-mod', 'block-no-situ', 'block-rounding'), dynamic=True if mode in ('block-dynamic', 'block-inline') else None)
    records = []
    try:
        for length in map(int, os.environ.get('MOSS_SHAPE_LENGTHS', '4408,4119,1075,1047,257,129,65,33,17,18,4408').split(',')):
            part = x[:, :length].clone()
            positions = pe[:, :length].clone()
            work = part.clone()
            # Restore the ordinary attention/Kitchen only while obtaining the reference.
            selected = (moss.attention, moss.comfy.quant_ops.ck.rms_adaln, moss.comfy.quant_ops.ck.rms_rope_split_half, comfy.ops.scaled_dot_product_attention)
            moss.attention, moss.comfy.quant_ops.ck.rms_adaln, moss.comfy.quant_ops.ck.rms_rope_split_half, comfy.ops.scaled_dot_product_attention = original_attention, original_adaln, original_rope, original_sdpa
            reference = block(part.clone(), mod, positions, {}).float().cpu()
            moss.attention, moss.comfy.quant_ops.ck.rms_adaln, moss.comfy.quant_ops.ck.rms_rope_split_half, comfy.ops.scaled_dot_product_attention = selected
            started = time.perf_counter()
            actual = candidate(work, mod, positions, {})
            torch.hpu.synchronize()
            cold = time.perf_counter() - started
            delta = actual.float().cpu() - reference
            timings = []
            for _ in range(5):
                torch.hpu.synchronize()
                started = time.perf_counter()
                for _ in range(5):
                    work.copy_(part)
                    candidate(work, mod, positions, {})
                torch.hpu.synchronize()
                timings.append((time.perf_counter() - started) / 5)
            record = {'length': length, 'cold': cold, 'median': statistics.median(timings), 'max_abs': float(delta.abs().max()),
                      'relative_rmse': float(delta.square().mean().sqrt() / reference.square().mean().sqrt()), 'finite': bool(torch.isfinite(delta).all())}
            records.append(record)
            print(mode, record, flush=True)
    except Exception:
        records.append({'exception': traceback.format_exc()})
        print(mode, records[-1], flush=True)
    results[mode] = records
    (root / os.environ.get('MOSS_SHAPE_RESULT', 'dynamic-shapes.json')).write_text(json.dumps(results, indent=2))
print('fallback', debug._get_fallback_op_count(), flush=True)
