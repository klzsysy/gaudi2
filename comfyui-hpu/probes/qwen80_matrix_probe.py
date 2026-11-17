"""Real FP8 PE weights, rotating beyond SRAM, with both physical layouts."""
import json
from pathlib import Path
import statistics
import sys
import time
import torch
import habana_frameworks.torch
from safetensors import safe_open
from comfy.quant_ops import hpu_fp8_linear


def forward(inputs, weights, scales, transposed):
    outputs = []
    for x, weight, weight_scale in zip(inputs, weights, scales):
        scale = x.abs().amax(-1, keepdim=True).float().clamp_min(1e-12) / 240.
        quantized = torch.ops.hpu.cast_to_fp8_v2(x, scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
        outputs.append(torch.ops.hpu.fp8_gemm_v2(quantized, False, weight, not transposed, None,
                                               x.dtype, scale, weight_scale, None, False))
    return outputs


torch.set_grad_enabled(False)
torch.set_num_threads(32)
torch._dynamo.config.recompile_limit = 32
root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
path = '/home/sonny/ComfyUI-Models/models/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.fp8_from_int8_fp32.safetensors'
with safe_open(path, framework='pt', device='cpu') as tensors:
    names = [key for key in tensors.keys() if key.endswith('mlp.gate_proj.weight')][:3]
    assert len(names) == 3, names
    weights = [tensors.get_tensor(name).to('hpu') for name in names]
    scales = [tensors.get_tensor(name.removesuffix('weight') + 'weight_scale').to('hpu') for name in names]
transposed_weights = [weight.t().contiguous() for weight in weights]
compiled = torch.compile(forward, backend='hpu_backend', fullgraph=True, dynamic=False,
                         options={'force_static_compile': True})
rows = []
for count in (1, 2, 4, 8, 16, 32):
    inputs = [torch.randn(count, weight.shape[1], device='hpu', dtype=torch.bfloat16) for weight in weights]
    reference = compiled(inputs, weights, scales, False)
    reference = [value.cpu() for value in reference]
    for transposed in (False, True):
        selected = transposed_weights if transposed else weights
        actual = [value.cpu() for value in compiled(inputs, selected, scales, transposed)]
        elapsed = []
        for repeat in range(6):
            torch.hpu.synchronize()
            start = time.perf_counter()
            for _ in range(40):
                compiled(inputs, selected, scales, transposed)
            torch.hpu.synchronize()
            elapsed.append((time.perf_counter() - start) / 40)
        seconds = statistics.median(elapsed[1:])
        elements = sum(weight.numel() for weight in weights)
        row = dict(rows=count, transposed=transposed, seconds=seconds, weight_bytes=elements,
                   weight_tbps=elements / seconds / 1e12, useful_tflops=2 * count * elements / seconds / 1e12,
                   exact=all(torch.equal(left, right) for left, right in zip(actual, reference)))
        rows.append(row)
        print(row, flush=True)
        (root / 'results.json').write_text(json.dumps(dict(weights=names, results=rows), indent=2))
