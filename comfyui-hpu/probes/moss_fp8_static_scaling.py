"""Measure calibrated activation scales with actual Moody Linear inputs."""
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
import comfy.quant_ops
import comfy.sd
import comfy.model_management
from comfy.ldm.moss_image import model as moss
from habana_frameworks.torch.utils.debug import _get_fallback_op_count

torch.set_grad_enabled(False)
torch.set_num_threads(32)
root = Path(os.environ['MOSS_DEEP_ROOT'])
patcher = comfy.sd.load_diffusion_model('/home/sonny/ComfyUI-Models/models/diffusion_models/Moody-Moss-Mix-V4.1_00001__clean-fp8.safetensors')
comfy.model_management.load_models_gpu([patcher])
model = patcher.model.diffusion_model
original = comfy.quant_ops.hpu_fp8_linear
captured = []

def capture(x, w, sw, bias=None, input_scale=None):
    captured.append((x, w, sw, bias))
    return original(x, w, sw, bias, input_scale)

def static_tensor(x, w, sw, bias, scale):
    return original(x, w, sw, bias, scale)

def static_scalar(x, w, sw, bias, scale):
    quantized = torch.ops.hpu.cast_to_fp8_v2(x.reshape(-1, x.shape[-1]), 1 / scale, False, False, torch.float8_e4m3fn)[0]
    output = torch.ops.hpu.fp8_gemm_v2(quantized, False, w, True, None, x.dtype, scale, sw, bias, False)
    return output.reshape(*x.shape[:-1], w.shape[0])

def measure(function, values):
    for _ in range(3):
        output = function(*values)
    torch.hpu.synchronize()
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(10):
            output = function(*values)
        torch.hpu.synchronize()
        samples.append((time.perf_counter() - start) / 10)
    return output, statistics.median(samples)

records = []
for source in sorted((root / 'baseline/captures').glob('input-*-0-0.pt')):
    data = torch.load(source, weights_only=True)
    block = model.layers[data['layer']]
    x, pe = data['x'].to('hpu'), data['pe'].to('hpu')
    mod = tuple(value.to('hpu') for value in data['mod'])
    comfy.quant_ops.hpu_fp8_linear = capture
    captured.clear()
    block(x, mod, pe, {'_moss_hpu_blockwise_rope': data['blockwise']})
    comfy.quant_ops.hpu_fp8_linear = original
    for name, values in zip(('qkv', 'o', 'ffn_up_gate', 'ffn_down'), captured):
        activation, weight, weight_scale, bias = values
        scale = activation.abs().amax().float().clamp_min(1e-12) / 240.0
        scalar_scale, scalar_weight = scale.item(), weight_scale.item()
        reference = None
        for mode, function, inputs in [('row', original, values),
                                      ('static-tensor', static_tensor, (*values, scale)),
                                      ('static-scalar', static_scalar, (activation, weight, scalar_weight, bias, scalar_scale))]:
            torch._dynamo.reset()
            compiled = torch.compile(function, backend='hpu_backend', dynamic=False)
            output, seconds = measure(compiled, inputs)
            if reference is None:
                reference = output.float()
            record = dict(source=source.name, name=name, mode=mode, seconds=seconds,
                          scale=scalar_scale, rrmse=((output.float()-reference).norm()/reference.norm()).item(),
                          finite=output.isfinite().all().item(), fallback=_get_fallback_op_count())
            records.append(record)
            (root / 'static-scaling.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(record), flush=True)
