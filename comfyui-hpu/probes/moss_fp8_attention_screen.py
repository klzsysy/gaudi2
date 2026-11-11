"""Screen vendor FP8 SDPA on captured MOSS BF16 activations, including quantization."""
import json
import os
from pathlib import Path
import statistics
import time

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.use_pytorch_cross_attention = True
import comfy.ops
from comfy.ldm.moss_image import model as moss
from habana_frameworks.torch.hpex.kernels.Fp8FusedSDPA import Fp8FusedSDPA
from habana_frameworks.torch.utils.debug import _get_fallback_op_count

torch.set_grad_enabled(False)
torch.set_num_threads(32)
root = Path(os.environ['MOSS_SYSTEM_ROOT'])
captures = root.parent / 'performance-2026-11-11-completion/captures'
block = moss.Block(3072, 8192, 24, fused_qkv=True, dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
block.load_state_dict(torch.load(captures / 'weights-0.pt', weights_only=True))
block.to('hpu')
original = comfy.ops.scaled_dot_product_attention
values = []
def capture(q, k, v, *args, **kwargs):
    values.append((q, k, v))
    return original(q, k, v, *args, **kwargs)
comfy.ops.scaled_dot_product_attention = capture
args.disable_comfy_compiler = True
data = torch.load(captures / 'input-307-0-0.pt', weights_only=True)
x, mod, pe = data['args']
block(x.to('hpu'), tuple(value.to('hpu') for value in mod), pe.to('hpu'), {})
comfy.ops.scaled_dot_product_attention = original
q, k, v = values[0]

def baseline(q, k, v):
    return torch.cat([original(q[:, i:i+1], k[:, i:i+1], v[:, i:i+1]) for i in range(q.shape[1])], dim=1)

def fp8(q, k, v, group=24):
    scales = [value.abs().amax().float().clamp_min(1e-12) / 240.0 for value in (q, k, v)]
    quantized = [torch.ops.hpu.cast_to_fp8_v2(value, scale.reciprocal(), False, False, torch.float8_e4m3fn)[0] for value, scale in zip((q, k, v), scales)]
    softmax_scale = torch.full((), 240.0, device=q.device, dtype=torch.float32)
    outputs = []
    for index in range(0, q.shape[1], group):
        query, key, value = [tensor[:, index:index+group] for tensor in quantized]
        output, _, _ = Fp8FusedSDPA.apply(query, key, value, None, 0.0, False, None, 'fp32',
                                         *scales, softmax_scale, None, softmax_scale.reciprocal(), False, False)
        outputs.append(output)
    return torch.cat(outputs, dim=1)

def fp8_gemm(q, k, v):
    scales = [value.abs().amax().float().clamp_min(1e-12) / 240.0 for value in (q, k, v)]
    quantized = [torch.ops.hpu.cast_to_fp8_v2(value, scale.reciprocal(), False, False, torch.float8_e4m3fn)[0] for value, scale in zip((q, k, v), scales)]
    probability_scale = torch.full((), 240.0, device=q.device, dtype=torch.float32)
    outputs = []
    for index in range(q.shape[1]):
        query, key, value = [tensor[:, index:index+1] for tensor in quantized]
        scores = torch.ops.hpu.fp8_gemm_v2(query, False, key, True, None, q.dtype, scales[0] / q.shape[-1]**.5, scales[1], None, False)
        probabilities = scores.softmax(-1)
        probabilities = torch.ops.hpu.cast_to_fp8_v2(probabilities, probability_scale, False, False, torch.float8_e4m3fn)[0]
        output = torch.ops.hpu.fp8_gemm_v2(probabilities, False, value, False, None, q.dtype, probability_scale.reciprocal(), scales[2], None, False)
        outputs.append(output)
    return torch.cat(outputs, dim=1)

results = []
reference = None
for name, function in [('bf16-heads', baseline), ('fp8-all', fp8), ('fp8-heads', lambda q, k, v: fp8(q, k, v, 1)), ('fp8-gemm-heads', fp8_gemm)]:
    try:
        compiled = torch.compile(function, backend='hpu_backend', dynamic=False)
        for _ in range(3):
            output = compiled(q, k, v)
        torch.hpu.synchronize()
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(5):
                output = compiled(q, k, v)
            torch.hpu.synchronize()
            samples.append((time.perf_counter()-start)/5)
        value = output.float()
        if reference is None:
            reference = value.clone()
        entry = {'name': name, 'seconds': statistics.median(samples),
                 'rrmse': ((value-reference).norm()/reference.norm()).item(), 'finite': value.isfinite().all().item(),
                 'fallback': _get_fallback_op_count()}
    except (RuntimeError, TypeError) as exc:
        entry = {'name': name, 'error': str(exc)}
    results.append(entry)
    (root / 'attention-screen-replay.json').write_text(json.dumps(results, indent=2))
    print(entry, flush=True)
