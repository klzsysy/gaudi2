"""Screen FP8 attention with fused softmax output quantization on captured inputs."""
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
from habana_frameworks.torch.utils.debug import _get_fallback_op_count

torch.set_grad_enabled(False)
torch.set_num_threads(32)
root = Path(os.environ['MOSS_DEEP_ROOT'])
root.mkdir(parents=True, exist_ok=True)
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
probability_scale = float(os.environ.get('MOSS_PROBABILITY_SCALE', '128'))

def baseline(q, k, v):
    return torch.cat([original(q[:, i:i+1], k[:, i:i+1], v[:, i:i+1]) for i in range(q.shape[1])], dim=1)

def attention(q, k, v, group, chunk, fused, qk_fp8):
    scales = [value.abs().amax().float().clamp_min(1e-12) / 240.0 for value in (q, k, v)]
    quantized = [torch.ops.hpu.cast_to_fp8_v2(value, scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
                 for value, scale in zip((q, k, v), scales)]
    p_inverse = torch.full((), 1 / probability_scale, device=q.device, dtype=torch.float32)
    outputs = []
    for head in range(0, q.shape[1], group):
        tiles = []
        key, value = [tensor[:, head:head+group] for tensor in quantized[1:]]
        for row in range(0, q.shape[-2], chunk):
            if qk_fp8:
                query = quantized[0][:, head:head+group, row:row+chunk]
                scores = torch.ops.hpu.fp8_gemm_v2(query, False, key, True, None, q.dtype,
                                                 scales[0] / q.shape[-1]**.5, scales[1], None, False)
            else:
                scores = (q[:, head:head+group, row:row+chunk] @ k[:, head:head+group].transpose(-1, -2)) / q.shape[-1]**.5
            if fused:
                probabilities = torch.ops.hpu.softmax_fp8.Scalar_scales(scores, -1, 1.0, probability_scale)
            else:
                probabilities = torch.ops.hpu.cast_to_fp8_v2(scores.softmax(-1), probability_scale, False, False, torch.float8_e4m3fn)[0]
            output = torch.ops.hpu.fp8_gemm_v2(probabilities, False, value, False, None, q.dtype,
                                             p_inverse, scales[2], None, False)
            tiles.append(output)
        outputs.append(torch.cat(tiles, dim=-2) if len(tiles) > 1 else tiles[0])
    return torch.cat(outputs, dim=1)

records = []
reference = None
cases = os.environ.get('MOSS_ATTENTION_CASES', 'baseline,split-1-8192,fused-1-8192,fused-2-8192,fused-4-8192,fused-8-8192,fused-24-8192,fused-1-1024,fused-2-1024,fused-4-512,av-1-8192').split(',')
for name in cases:
    torch._dynamo.reset()
    if name == 'baseline':
        function = baseline
    else:
        mode, group, chunk = name.split('-')
        group, chunk = int(group), int(chunk)
        def function(q, k, v):
            return attention(q, k, v, group, chunk, mode != 'split', mode != 'av')
    compiled = torch.compile(function, backend='hpu_backend', dynamic=False, options={'force_static_compile': True})
    for _ in range(3):
        output = compiled(q, k, v)
    torch.hpu.synchronize()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(10):
            output = compiled(q, k, v)
        torch.hpu.synchronize()
        samples.append((time.perf_counter()-start)/10)
    value = output.float()
    if reference is None:
        reference = value.clone()
    record = dict(name=name, seconds=statistics.median(samples), samples=samples,
                  rrmse=((value-reference).norm()/reference.norm()).item(),
                  finite=value.isfinite().all().item(), fallback=_get_fallback_op_count())
    records.append(record)
    (root / os.environ.get('MOSS_ATTENTION_OUTPUT', 'attention-fusion.json')).write_text(json.dumps(records, indent=2))
    print(json.dumps(record), flush=True)
