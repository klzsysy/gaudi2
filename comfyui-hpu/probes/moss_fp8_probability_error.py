"""Compare native fused FP8 softmax with explicit FP32/FP8 CPU references."""
import json
import os
from pathlib import Path
import torch
import habana_frameworks.torch.core
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.use_pytorch_cross_attention = True
from comfy.ldm.modules.attention import _hpu_fp8_attention_av

torch.set_grad_enabled(False)
torch.set_num_threads(32)
torch.manual_seed(72)
q, k, v = torch.randn(1, 4403, 3 * 128, dtype=torch.bfloat16).to('hpu').chunk(3, dim=-1)
q, k, v = [value.unsqueeze(1) for value in (q, k, v)]
scale = 128 ** -.25
scores = (q[0, 0] * scale) @ (k[0, 0] * scale).T
probabilities = torch.ops.hpu.softmax_fp8.Scalar_scales(scores, -1, 1.0, 128.0)
p = scores.cpu().float().softmax(-1)
pq = (p * 128).to(torch.float8_e4m3fn).float() / 128
native_p = probabilities.float().cpu() / 128
vc = v[0, 0].cpu().float()
vs = vc.abs().amax() / 240
vq = (vc / vs).to(torch.float8_e4m3fn).float() * vs
reference = p @ vc
quantized_reference = pq @ vq
native_reference = native_p @ vq
actual = torch.compile(_hpu_fp8_attention_av, backend='hpu_backend', dynamic=False)(q, k, v)[0, 0].cpu().float()
def error(value, reference):
    return ((value-reference).norm()/reference.norm()).item()
result = dict(p_max=p.max().item(), p_quant_error=error(pq,p), native_p_error=error(native_p,p),
              native_p_vs_quant=error(native_p,pq), quantized_output_error=error(quantized_reference,reference),
              native_reference_error=error(native_reference,reference), actual_error=error(actual,reference),
              actual_vs_quant=error(actual,quantized_reference), actual_vs_native_reference=error(actual,native_reference),
              quantized_zero_fraction=(pq == 0).float().mean().item(), native_zero_fraction=(native_p == 0).float().mean().item())
print(json.dumps(result, indent=2))
(Path(os.environ['MOSS_DEEP_ROOT']) / 'probability-error.json').write_text(json.dumps(result, indent=2))
