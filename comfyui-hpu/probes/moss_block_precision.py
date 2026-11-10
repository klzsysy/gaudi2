"""Locate the first compiled-block divergence on captured real BF16 inputs."""
import json
import os
from pathlib import Path

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
moss.attention = moss._attention
def fp16_wanted(x):
    return x.is_cuda and x.dtype == torch.float16 and getattr(torch.backends.cuda.matmul, 'allow_fp16_accumulation', False)
comfy.ops._fp16_linear_wanted = fp16_wanted
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
mask = source['mask'].to('hpu') if source['mask'] is not None else None


def prefix(layer, x, mod, pe, stage):
    y = moss.modulate(x, layer.norm1, mod, 0, None)
    if stage == 0:
        return y
    qkv = layer.proj['qkv'](y)
    if stage == 1:
        return qkv
    q, k, v = qkv.chunk(3, dim=-1)
    a = moss.attention(q, k, v, layer.q_norm, layer.k_norm, pe, layer.num_heads, layer.comfy_attention, {}, mask=mask)
    if stage == 2:
        return a
    o = layer.proj['o'](a)
    if stage == 3:
        return o
    x = moss.residual(x, o, mod, 2, None)
    if stage == 4:
        return x
    y = moss.modulate(x, layer.norm2, mod, 3, None)
    if stage == 5:
        return y
    y = layer.proj['ffn_up_gate'](y)
    if stage == 6:
        return y
    if stage == 7:
        return comfy.ops._eager_input_act(y, 'situ')
    y = comfy.ops.linear_input_act(layer.proj['ffn_down'], y, 'situ')
    if stage == 8:
        return y
    return moss.residual(x, y, mod, 5, None)


def compare(actual, expected):
    actual = actual.detach().cpu().float()
    expected = expected.detach().cpu().float()
    delta = actual - expected
    return {'max_abs': float(delta.abs().max()), 'mean_abs': float(delta.abs().mean()),
            'relative_rmse': float(delta.square().mean().sqrt() / expected.square().mean().sqrt()),
            'changed_fraction': float((delta != 0).float().mean()), 'finite': bool(torch.isfinite(actual).all())}


names = ['modulation1', 'qkv', 'attention', 'out_projection', 'residual1', 'modulation2', 'ffn_up_gate', 'situ', 'ffn_down', 'residual2']
results = {}
for stage, name in enumerate(names):
    torch._dynamo.reset()
    expected = prefix(block, x.clone(), mod, pe, stage).cpu()
    compiled = torch.compile(prefix, backend='hpu_backend', fullgraph=True, dynamic=False)
    actual = compiled(block, x.clone(), mod, pe, stage)
    result = compare(actual, expected)
    results[name] = result
    print(name, result, flush=True)
    (root / 'prefix-errors.json').write_text(json.dumps(results, indent=2))

# Isolate pointwise operations from upstream errors, using identical real inputs.
attn_projection = prefix(block, x.clone(), mod, pe, 3)
ffn_input = prefix(block, x.clone(), mod, pe, 6)
tests = [
    ('modulation', moss.modulated_rms_norm, (x, block.norm1, mod[1], mod[0])),
    ('residual', moss.gated_residual, (x.clone(), attn_projection, mod[2])),
    ('situ', comfy.ops.INPUT_ACT_EAGER['situ'], (ffn_input,)),
]
isolated = {}
for name, function, values in tests:
    torch._dynamo.reset()
    def copied():
        return tuple(v.clone() if isinstance(v, torch.Tensor) else v for v in values)
    expected = function(*copied()).cpu()
    actual = torch.compile(function, backend='hpu_backend', fullgraph=True, dynamic=False)(*copied()).cpu()
    record = compare(actual, expected)
    # FP64 pointwise reference; no attention or GEMM is approximated here.
    if name == 'residual':
        a, b, c = (v.cpu().double() for v in values)
        reference = torch.addcmul(a, b, c)
    elif name == 'situ':
        reference = function(values[0].cpu().double())
    else:
        a = values[0].cpu().double()
        w = block.norm1.weight.cpu().double()
        scale, shift = values[2].cpu().double(), values[3].cpu().double()
        reference = torch.nn.functional.rms_norm(a, (a.shape[-1],), eps=block.norm1.eps) * w * (1 + scale) + shift
    record['eager_vs_fp64'] = compare(expected, reference)
    record['compiled_vs_fp64'] = compare(actual, reference)
    isolated[name] = record
    print('ISOLATED', name, record, flush=True)
(root / 'isolated-errors.json').write_text(json.dumps(isolated, indent=2))
print('fallback', debug._get_fallback_op_count(), flush=True)
