"""Separate long-sequence compiled probability, value and AV reference errors."""
import json
from pathlib import Path
import sys

import torch
from comfy.cli_args import args
args.use_pytorch_cross_attention = True
from comfy.ldm.modules.attention import _hpu_fp8_attention_av


def components(q, k, v):
    value = v[0, 0]
    value_scale = value.abs().amax().float().clamp_min(1e-12) / 240.0
    quantized_value = torch.ops.hpu.cast_to_fp8_v2(value, value_scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
    scale = q.shape[-1] ** -0.25
    scores = (q[0, 0] * scale) @ (k[0, 0] * scale).T
    probabilities = torch.ops.hpu.softmax_fp8.Scalar_scales(scores, -1, 1.0, 128.0)
    return probabilities, quantized_value, value_scale


torch.set_grad_enabled(False)
torch.set_num_threads(32)
records = []
for length in (4403, 8629, 16385):
    torch._dynamo.reset()
    torch.manual_seed(72)
    q, k, v = torch.randn(1, 1, length, 3 * 128, dtype=torch.bfloat16).to("hpu").chunk(3, dim=-1)
    actual = torch.compile(_hpu_fp8_attention_av, backend="hpu_backend", dynamic=False)(q, k, v).float().cpu()[0, 0]
    p, value, value_scale = torch.compile(components, backend="hpu_backend", dynamic=False)(q, k, v)
    p = p.cpu().float() / 128
    value = value.cpu().float() * value_scale.cpu()
    eager_p = components(q, k, v)[0].cpu().float() / 128
    cpu_v = v[0, 0].cpu().float()
    cpu_scale = cpu_v.abs().amax() / 240
    cpu_v = (cpu_v / cpu_scale).to(torch.float8_e4m3fn).float() * cpu_scale
    expected = p @ value
    record = dict(length=length, av_rrmse=((actual - expected).norm() / expected.norm()).item(),
                  probability_compile_rrmse=((p - eager_p).norm() / eager_p.norm()).item(),
                  value_quantization_rrmse=((value - cpu_v).norm() / cpu_v.norm()).item(),
                  reference_rrmse=((actual - eager_p @ cpu_v).norm() / (eager_p @ cpu_v).norm()).item())
    records.append(record)
    Path(sys.argv[1]).write_text(json.dumps(records, indent=2))
    print(json.dumps(record), flush=True)
