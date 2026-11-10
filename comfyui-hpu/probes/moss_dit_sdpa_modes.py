"""Compare existing vendor SDPA modes on captured MOSS tensors and a real block."""
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
from habana_frameworks.torch.hpex.kernels import FusedSDPA


torch.set_grad_enabled(False)
torch.set_num_threads(16)
root = Path(os.environ['MOSS_PREVIOUS_BLOCKS'])
destination = Path(os.environ['MOSS_DIT_NEXT'])
state = torch.load(root / 'captures/weights-0.pt', weights_only=True)
data = torch.load(root / 'captures/input-0-0.pt', weights_only=True)
block = moss.Block(3072, 8192, 24, fused_qkv=True, dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
block.load_state_dict(state)
block.to('hpu')
x, mod, pe = data['args']
x, mod, pe = x.to('hpu'), tuple(v.to('hpu') for v in mod), pe.to('hpu')
baseline_sdpa = comfy.ops.scaled_dot_product_attention
captured = []
def capture(q, k, v, *values, **kwargs):
    captured.append((q.clone(), k.clone(), v.clone(), values, kwargs))
    return baseline_sdpa(q, k, v, *values, **kwargs)
comfy.ops.scaled_dot_product_attention = capture
args.disable_comfy_compiler = True
block(x.clone(), mod, pe, {})
comfy.ops.scaled_dot_product_attention = baseline_sdpa
args.disable_comfy_compiler = False
q, k, v, values, kwargs = captured[0]
fp64 = torch.load(root / 'fp64-0.pt', weights_only=True).double()


def measure(function):
    actual = function()
    torch.hpu.synchronize()
    for _ in range(3):
        function()
    torch.hpu.synchronize()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(10):
            function()
        torch.hpu.synchronize()
        samples.append((time.perf_counter() - start) / 10)
    return actual.cpu().double(), {'median_ms': statistics.median(samples) * 1000, 'samples_ms': [s * 1000 for s in samples]}


def error(actual, expected):
    return {'rrmse': float((actual-expected).square().mean().sqrt() / expected.square().mean().sqrt()),
            'max_abs': float((actual-expected).abs().max()), 'finite': bool(actual.isfinite().all())}


results = []
baseline_block = None
baseline_attention = None
cases = [('math', False), ('None', False), ('None', True), ('fast', False), ('fast', True), ('fp32', False)]
if os.environ.get('MOSS_CENTER_KEYS') == '1':
    cases = [('math', False), ('center-bf16', False), ('center-fp32', False)]
for mode, recompute in cases:
    torch._dynamo.reset()
    moss._compiled_block_ops.cache_clear()
    def native(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        if mode == 'center-bf16':
            k = k - k.mean(dim=-2, keepdim=True)
        elif mode == 'center-fp32':
            work = k.float()
            k = (work - work.mean(dim=-2, keepdim=True)).to(k.dtype)
        softmax = 'fast' if mode.startswith('center') else mode
        return FusedSDPA.apply(q, k, v, attn_mask, dropout_p, is_causal, scale, softmax, recompute)
    function = baseline_sdpa if mode == 'math' else native
    comfy.ops.scaled_dot_product_attention = function
    compiled = torch.compile(function, backend='hpu_backend', dynamic=False, options={'force_static_compile': True})
    try:
        actual_attention, at = measure(lambda: compiled(q, k, v, *values, **kwargs))
        actual_block, bt = measure(lambda: block(x.clone(), mod, pe, {}))
        if mode == 'math':
            baseline_attention, baseline_block = actual_attention, actual_block
        record = {'mode': mode, 'recompute': recompute, 'attention': at, 'block': bt,
                  'attention_error': error(actual_attention, baseline_attention), 'block_error': error(actual_block, baseline_block),
                  'block_fp64_error': error(actual_block, fp64), 'fallback': debug._get_fallback_op_count()}
    except Exception as exc:
        record = {'mode': mode, 'recompute': recompute, 'error': repr(exc)}
    results.append(record)
    (destination / os.environ.get('MOSS_SDPA_RESULT', 'sdpa-modes.json')).write_text(json.dumps(results, indent=2))
    print(json.dumps(record), flush=True)
