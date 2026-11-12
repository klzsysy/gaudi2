"""Validate attention on Moody block captures across layers, steps and conditions."""
import functools
import json
import os
from pathlib import Path
import statistics
import time

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.use_pytorch_cross_attention = True
args.disable_comfy_compiler = True
import comfy.ops
import comfy.sd
import comfy.model_management
from comfy.ldm.moss_image import model as moss
from habana_frameworks.torch.utils.debug import _get_fallback_op_count
from moss_fp8_attention_candidate import fp8_attention, fp8_attention_variant

torch.set_grad_enabled(False)
torch.set_num_threads(32)
root = Path(os.environ['MOSS_DEEP_ROOT'])
patcher = comfy.sd.load_diffusion_model('/home/sonny/ComfyUI-Models/models/diffusion_models/Moody-Moss-Mix-V4.1_00001__clean-fp8.safetensors')
comfy.model_management.load_models_gpu([patcher])
model = patcher.model.diffusion_model
original = comfy.ops.scaled_dot_product_attention
values = []

def capture(q, k, v, *args, **kwargs):
    values.append((q, k, v))
    return original(q, k, v, *args, **kwargs)

def baseline(q, k, v):
    return torch.cat([original(q[:, i:i+1], k[:, i:i+1], v[:, i:i+1]) for i in range(q.shape[1])], dim=1)

def measure(function, values):
    for _ in range(3):
        output = function(*values)
    torch.hpu.synchronize()
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(5):
            output = function(*values)
        torch.hpu.synchronize()
        samples.append((time.perf_counter() - start) / 5)
    return output, statistics.median(samples)

compiled_baseline = torch.compile(baseline, backend='hpu_backend', dynamic=False)
candidate = fp8_attention
if os.environ.get('MOSS_DEEP_ATTENTION_MODE'):
    candidate = functools.partial(fp8_attention_variant, mode=os.environ['MOSS_DEEP_ATTENTION_MODE'])
compiled_candidate = torch.compile(candidate, backend='hpu_backend', dynamic=False)
records = []
for source in sorted((root / 'baseline/captures').glob('input-*.pt')):
    data = torch.load(source, weights_only=True)
    block = model.layers[data['layer']]
    x, pe = data['x'].to('hpu'), data['pe'].to('hpu')
    mod = tuple(value.to('hpu') for value in data['mod'])
    comfy.ops.scaled_dot_product_attention = capture
    values.clear()
    block._forward_attention(moss.modulate(x, block.norm1, mod, 0, None), pe,
                             {'_moss_hpu_blockwise_rope': data['blockwise']})
    comfy.ops.scaled_dot_product_attention = original
    q, k, v = values[0]
    expected, base_time = measure(compiled_baseline, (q, k, v))
    actual, candidate_time = measure(compiled_candidate, (q, k, v))
    # FP32 SDPA provides a separate reference for accumulated attention error.
    reference = baseline(q.float(), k.float(), v.float())
    difference = actual.float() - expected.float()
    record = dict(source=source.name, baseline_s=base_time, candidate_s=candidate_time,
                  rrmse=(difference.norm()/expected.float().norm()).item(),
                  fp32_baseline_rrmse=((expected.float()-reference).norm()/reference.norm()).item(),
                  fp32_candidate_rrmse=((actual.float()-reference).norm()/reference.norm()).item(),
                  finite=actual.isfinite().all().item(), fallback=_get_fallback_op_count())
    records.append(record)
    (root / os.environ.get('MOSS_ATTENTION_VALIDATION_OUTPUT', 'attention-moody-validation.json')).write_text(json.dumps(records, indent=2))
    print(json.dumps(record), flush=True)
