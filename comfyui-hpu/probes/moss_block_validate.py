"""Validate compiled regions across captured layers/timesteps and FP64 GEMM rows."""
import json
import os
from pathlib import Path
import time

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True
import comfy.ops
from comfy.ldm.moss_image import model as moss
import moss_block_candidate as candidate
import habana_frameworks.torch.utils.debug as debug

torch.set_grad_enabled(False)
torch.set_num_threads(16)
comfy.ops._fp16_linear_wanted = candidate.fp16_wanted
root = Path(os.environ['MOSS_BLOCK_ROOT'])
capture = root / 'captures'
results = []


def error(actual, expected):
    actual, expected = actual.cpu().double(), expected.cpu().double()
    delta = actual - expected
    return {'finite': bool(torch.isfinite(actual).all()), 'max_abs': float(delta.abs().max()), 'mean_abs': float(delta.abs().mean()),
            'relative_rmse': float(delta.square().mean().sqrt() / expected.square().mean().sqrt())}


def linear(layer, x):
    return layer(x)


compiled_linear = torch.compile(linear, backend='hpu_backend', fullgraph=True, dynamic=False)
for layer in (0, 20, 41):
    state = torch.load(capture / f'weights-{layer}.pt', weights_only=True)
    block = moss.Block(state['norm1.weight'].numel(), state['proj.ffn_up_gate.weight'].shape[0] // 2, 24, fused_qkv=True, dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
    block.load_state_dict(state)
    block.to('hpu')
    for step in (0, 12, 24):
        data = torch.load(capture / f'input-{step}-{layer}.pt', weights_only=True)
        x = data['args'][0].to('hpu')
        mod = tuple(t.to('hpu') for t in data['args'][1])
        pe = data['args'][2].to('hpu')
        expected = block(x.clone(), mod, pe, {}).cpu()
        actual = candidate.forward(block, x.clone(), mod, pe, {}).cpu()
        record = {'layer': layer, 'step': step, 'block_error': error(actual, expected)}
        reference_path = root / (f'fp64-{layer}.pt' if step == 0 else f'fp64-{step}-{layer}.pt')
        if reference_path.exists():
            high_precision = torch.load(reference_path, weights_only=True)
            record['eager_block_vs_fp64'] = error(expected, high_precision)
            record['compiled_block_vs_fp64'] = error(actual, high_precision)
        if step == 0:
            y = moss.modulate(x, block.norm1, mod, 0, None)
            original = block.proj['qkv'](y).cpu()
            compiled = compiled_linear(block.proj['qkv'], y).cpu()
            # Include rows with differences and evenly spaced rows; each row's GEMM is independent.
            changed = (original != compiled).any(dim=-1)[0].nonzero().flatten().tolist()
            rows = sorted(set(changed[:8] + list(range(0, y.shape[1], max(1, y.shape[1] // 8)))))
            y_cpu = y.cpu()[0, rows].double()
            reference = torch.nn.functional.linear(y_cpu, state['proj.qkv.weight'].double())
            record['linear_rows'] = rows
            record['eager_linear_vs_fp64'] = error(original[0, rows], reference)
            record['compiled_linear_vs_fp64'] = error(compiled[0, rows], reference)
            record['different_linear_fraction'] = float((original != compiled).float().mean())
        results.append(record)
        (root / 'validation.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(record), flush=True)
print('fallback', debug._get_fallback_op_count(), flush=True)
