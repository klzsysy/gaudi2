"""Screen long-sequence FP8 probability quantization on the HPU."""
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from comfy.cli_args import args
args.use_pytorch_cross_attention = True
import comfy.ops
from comfy.ldm.modules.attention import _hpu_fp8_attention_av


def row_probability(q, k, v):
    query, key, value = q[0, 0], k[0, 0], v[0, 0]
    value_scale = value.abs().amax().float().clamp_min(1e-12) / 240.0
    quantized_value = torch.ops.hpu.cast_to_fp8_v2(value, value_scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
    scale = q.shape[-1] ** -0.25
    scores = (query * scale) @ (key * scale).transpose(-1, -2)
    probabilities = scores.softmax(-1)
    probability_scale = probabilities.amax(-1, keepdim=True).float().clamp_min(1e-12) / 240.0
    quantized_probability = torch.ops.hpu.cast_to_fp8_v2(probabilities, probability_scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
    output = torch.ops.hpu.fp8_gemm_v2(quantized_probability, False, quantized_value, False, None, q.dtype,
                                    probability_scale, value_scale, None, False)
    return output.unsqueeze(0).unsqueeze(0)


torch.set_grad_enabled(False)
torch.set_num_threads(32)
records = []
for length in (8193, 8629, 16385):
    torch.manual_seed(72)
    q, k, v = torch.randn(1, 1, length, 3 * 128, dtype=torch.bfloat16).to("hpu").chunk(3, dim=-1)
    reference = comfy.ops.scaled_dot_product_attention(q, k, v).float()
    for name, function in (("sdpa", comfy.ops.scaled_dot_product_attention), ("fixed", _hpu_fp8_attention_av), ("row", row_probability)):
        torch._dynamo.reset()
        compiled = torch.compile(function, backend="hpu_backend", dynamic=False)
        for _ in range(3):
            output = compiled(q, k, v)
        torch.hpu.synchronize()
        durations = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(10):
                output = compiled(q, k, v)
            torch.hpu.synchronize()
            durations.append((time.perf_counter() - start) / 10)
        record = dict(length=length, mode=name, seconds=statistics.median(durations),
                      rrmse=((output.float() - reference).norm() / reference.norm()).item(),
                      finite=bool(output.isfinite().all()))
        records.append(record)
        Path(sys.argv[1]).write_text(json.dumps(records, indent=2))
        print(json.dumps(record), flush=True)
