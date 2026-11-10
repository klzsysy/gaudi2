"""Compare block partitions that retain eager modulation and residual rounding."""
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
import habana_frameworks.torch.utils.debug as debug

torch.set_grad_enabled(False)
torch.set_num_threads(16)
root = Path(os.environ['MOSS_BLOCK_ROOT'])
root.mkdir(parents=True, exist_ok=True)
source = torch.load(os.environ['MOSS_BLOCK_CAPTURE'], weights_only=True)
state = source['state']
block = moss.Block(state['norm1.weight'].numel(), state['proj.ffn_up_gate.weight'].shape[0] // 2, source['heads'], fused_qkv=True, dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
block.load_state_dict(state)
block.to('hpu')
x = source['args'][0].to('hpu')
mod = tuple(t.to('hpu') for t in source['args'][1])
pe = source['args'][2].to('hpu')
work = x.clone()
original_modulation = moss.modulated_rms_norm
original_residual = moss.gated_residual
product_attention = moss.attention
def fp16_wanted(x):
    return x.is_cuda and x.dtype == torch.float16 and getattr(torch.backends.cuda.matmul, 'allow_fp16_accumulation', False)
comfy.ops._fp16_linear_wanted = fp16_wanted


def project_attention(layer, y, pe):
    q, k, v = layer.proj['qkv'](y).chunk(3, dim=-1)
    a = moss.attention(q, k, v, layer.q_norm, layer.k_norm, pe, layer.num_heads, layer.comfy_attention, {})
    return layer.proj['o'](a)


def feed_forward(layer, y):
    y = layer.proj['ffn_up_gate'](y)
    return comfy.ops.linear_input_act(layer.proj['ffn_down'], y, 'situ')


def split_forward(layer, x, mod, pe):
    y = moss.modulate(x, layer.norm1, mod, 0, None)
    x = moss.residual(x, compiled_projection(layer, y, pe), mod, 2, None)
    y = moss.modulate(x, layer.norm2, mod, 3, None)
    return moss.residual(x, compiled_ffn(layer, y), mod, 5, None)


def evaluate(fn):
    def run():
        work.copy_(x)
        return fn(block, work, mod, pe)
    start = time.perf_counter()
    actual = run()
    torch.hpu.synchronize()
    cold = time.perf_counter() - start
    actual_cpu = actual.float().cpu()
    for _ in range(3):
        run()
    torch.hpu.synchronize()
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(10):
            run()
        torch.hpu.synchronize()
        timings.append((time.perf_counter() - start) / 10)
    record = {'cold': cold, 'median': statistics.median(timings), 'samples': timings}
    if 'reference' in globals():
        delta = actual_cpu - reference
        record.update(max_abs=float(delta.abs().max()), mean_abs=float(delta.abs().mean()), relative_rmse=float(delta.square().mean().sqrt() / reference.square().mean().sqrt()), finite=bool(torch.isfinite(actual_cpu).all()))
    return actual_cpu, record


reference, baseline = evaluate(moss.Block.forward)
results = {'product': baseline}
for mode in os.environ.get('MOSS_REGION_MODES', 'whole,no-residual-fusion,preserve-rounding,split-static,split-auto').split(','):
    torch._dynamo.reset()
    moss.attention = moss._attention
    moss.modulated_rms_norm = original_modulation
    moss.gated_residual = original_residual
    if mode in ('no-residual-fusion', 'preserve-rounding'):
        moss.gated_residual = torch.compiler.disable(original_residual)
    if mode == 'preserve-rounding':
        moss.modulated_rms_norm = torch.compiler.disable(original_modulation)
    if mode.startswith('split'):
        dynamic = None if mode == 'split-auto' else False
        compiled_projection = torch.compile(project_attention, backend='hpu_backend', fullgraph=True, dynamic=dynamic)
        compiled_ffn = torch.compile(feed_forward, backend='hpu_backend', fullgraph=True, dynamic=dynamic)
        function = split_forward
    else:
        function = torch.compile(moss.Block.forward, backend='hpu_backend', fullgraph=mode == 'whole', dynamic=False)
    actual, result = evaluate(function)
    results[mode] = result
    print(mode, result, flush=True)
    (root / 'regions.json').write_text(json.dumps(results, indent=2))
print('fallback', debug._get_fallback_op_count(), flush=True)

if os.environ.get('MOSS_REGION_DIAG') == '1':
    moss.modulated_rms_norm, moss.gated_residual = original_modulation, original_residual
    moss.attention = moss._attention
    y1 = moss.modulate(x, block.norm1, mod, 0, None)
    after_attention = moss.residual(x.clone(), project_attention(block, y1, pe), mod, 2, None)
    y2 = moss.modulate(after_attention, block.norm2, mod, 3, None)
    def projection_prefix(layer, y, pe, stage):
        qkv = layer.proj['qkv'](y)
        if stage == 0:
            return qkv
        q, k, v = qkv.chunk(3, dim=-1)
        out = moss.attention(q, k, v, layer.q_norm, layer.k_norm, pe, layer.num_heads, layer.comfy_attention, {})
        return out if stage == 1 else layer.proj['o'](out)
    def ffn_prefix(layer, y, pe, stage):
        y = layer.proj['ffn_up_gate'](y)
        if stage == 0:
            return y
        return comfy.ops._eager_input_act(y, 'situ') if stage == 1 else comfy.ops.linear_input_act(layer.proj['ffn_down'], y, 'situ')
    diagnostic = {}
    for name, function, value in [('projection', projection_prefix, y1), ('ffn', ffn_prefix, y2)]:
        for stage in range(3):
            torch._dynamo.reset()
            expected = function(block, value, pe, stage).float().cpu()
            actual = torch.compile(function, backend='hpu_backend', fullgraph=True, dynamic=False)(block, value, pe, stage).float().cpu()
            delta = actual - expected
            record = {'max_abs': float(delta.abs().max()), 'mean_abs': float(delta.abs().mean()), 'relative_rmse': float(delta.square().mean().sqrt() / expected.square().mean().sqrt()), 'changed_fraction': float((delta != 0).float().mean())}
            diagnostic[name + str(stage)] = record
            print('REGION', name, stage, record, flush=True)
    (root / 'region-prefixes.json').write_text(json.dumps(diagnostic, indent=2))
