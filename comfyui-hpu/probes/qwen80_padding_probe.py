"""Measure useful FP8 M=1 bandwidth with non-power-of-two weight strides."""
import json
from pathlib import Path
import statistics
import sys
import time
import torch
import torch.nn.functional as F
import habana_frameworks.torch
from safetensors import safe_open


def forward(inputs, weights, scales):
    outputs = []
    for value, weight, weight_scale in zip(inputs, weights, scales):
        scale = value.abs().amax(-1, keepdim=True).float().clamp_min(1e-12) / 240.
        quantized = torch.ops.hpu.cast_to_fp8_v2(value, scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
        outputs.append(torch.ops.hpu.fp8_gemm_v2(quantized, False, weight, True, None,
                                               value.dtype, scale, weight_scale, None, False))
    return outputs


torch.set_grad_enabled(False)
torch.set_num_threads(32)
torch.manual_seed(42)
root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
path = '/home/sonny/ComfyUI-Models/models/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.fp8_from_int8_fp32.safetensors'
with safe_open(path, framework='pt', device='cpu') as tensors:
    names = [name for name in tensors.keys() if name.endswith('mlp.gate_proj.weight')][:3]
    weights = [tensors.get_tensor(name).to('hpu') for name in names]
    scales = [tensors.get_tensor(name.removesuffix('weight') + 'weight_scale').to('hpu') for name in names]
inputs = [torch.randn(1, weight.shape[1], device='hpu', dtype=torch.bfloat16) for weight in weights]
compiled = torch.compile(forward, backend='hpu_backend', fullgraph=True, dynamic=False,
                         options={'force_static_compile': True})
reference = [output.cpu().float() for output in compiled(inputs, weights, scales)]
useful_bytes = sum(weight.numel() for weight in weights)
rows = []
for padding in (0, 16, 32, 64, 128, 256, 512):
    padded_weights = weights if padding == 0 else [torch.ops.hpu.cast_to_fp8_v2(
        F.pad(weight.float(), (0, padding)), None, False, False, torch.float8_e4m3fn)[0] for weight in weights]
    padded_inputs = [F.pad(value, (0, padding)) for value in inputs]
    actual = [output.cpu().float() for output in compiled(padded_inputs, padded_weights, scales)]
    timings = []
    for repeat in range(6):
        torch.hpu.synchronize()
        start = time.perf_counter()
        for _ in range(40):
            compiled(padded_inputs, padded_weights, scales)
        torch.hpu.synchronize()
        timings.append((time.perf_counter() - start) / 40)
    seconds = statistics.median(timings[1:])
    row = dict(padding=padding, seconds=seconds, useful_weight_bytes=useful_bytes,
               physical_weight_bytes=sum(weight.numel() for weight in padded_weights),
               useful_weight_tbps=useful_bytes / seconds / 1e12,
               exact=all(torch.equal(left, right) for left, right in zip(actual, reference)),
               max_abs=max((left-right).abs().max().item() for left, right in zip(actual, reference)))
    rows.append(row)
    print(row, flush=True)
    (root / 'results.json').write_text(json.dumps(rows, indent=2))
