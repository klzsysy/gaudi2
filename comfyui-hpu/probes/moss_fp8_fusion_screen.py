"""Replay Moody blocks. Screen from f67e7f3f, or compare product/prototype after deployment."""
import json
import os
from pathlib import Path
import statistics
import time

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.use_pytorch_cross_attention = True
import comfy.sd
import comfy.ops
import comfy.quant_ops
import comfy.model_management
from habana_frameworks.torch.utils.debug import _get_fallback_op_count
import moss_fp8_fusion_candidate as candidate

torch.set_grad_enabled(False)
torch.set_num_threads(32)
root = Path(os.environ['MOSS_FUSION_ROOT'])
captures = root.parent / 'performance-2026-11-12-fp8-deep/baseline/captures'
patcher = comfy.sd.load_diffusion_model('/home/sonny/ComfyUI-Models/models/diffusion_models/Moody-Moss-Mix-V4.1_00001__clean-fp8.safetensors')
comfy.model_management.load_models_gpu([patcher])
model = patcher.model.diffusion_model
records = []


def measure(block, x, mod, pe, options):
    for _ in range(3):
        output = block(x.clone(), mod, pe, options)
    torch.hpu.synchronize()
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(10):
            output = block(x.clone(), mod, pe, options)
        torch.hpu.synchronize()
        samples.append((time.perf_counter() - start) / 10)
    return output.cpu().float(), statistics.median(samples)


sources = sorted(captures.glob(os.environ.get('MOSS_FUSION_INPUTS', 'input-307-12-*.pt')))
for source in sources:
    data = torch.load(source, weights_only=True)
    block = model.layers[data['layer']]
    x, pe = data['x'].to('hpu'), data['pe'].to('hpu')
    mod = tuple(value.to('hpu') for value in data['mod'])
    options = {'fp8_attention': True, '_moss_hpu_blockwise_rope': data['blockwise']}
    high_precision = None
    if os.environ.get('MOSS_FUSION_REFERENCE') == '1':
        shadow = candidate.moss.Block(3072, 8192, 24, fused_qkv=True, dtype=torch.float32,
                                      device='hpu', operations=comfy.ops.manual_cast)
        for name, target in shadow.named_parameters():
            value = block.get_parameter(name)
            if isinstance(value, comfy.quant_ops.QuantizedTensor):
                value = value.dequantize()
            target.copy_(value.float())
        candidate.install('baseline')
        high_precision = shadow(x.float(), tuple(value.float() for value in mod), pe, options).cpu().float()
        del shadow
    reference = None
    for mode in os.environ.get('MOSS_FUSION_MODES', 'baseline,regions,whole,rope,norm,whole-rope,whole-rope-norm').split(','):
        torch._dynamo.reset()
        candidate.install(mode)
        output, seconds = measure(block, x, mod, pe, options)
        if reference is None:
            reference = output
        record = dict(source=source.name, mode=mode, seconds=seconds,
                      rrmse=((output-reference).norm()/reference.norm()).item(),
                      finite=output.isfinite().all().item(), fallback=_get_fallback_op_count())
        if high_precision is not None:
            record['fp32_rrmse'] = ((output-high_precision).norm()/high_precision.norm()).item()
        records.append(record)
        (root / os.environ.get('MOSS_FUSION_OUTPUT', 'screen.json')).write_text(json.dumps(records, indent=2))
        print(json.dumps(record), flush=True)
        if os.environ.get('MOSS_FUSION_PROFILE') == '1':
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU]) as profile:
                block(x.clone(), mod, pe, options)
                torch.hpu.synchronize()
            profile.profiler.export_chrome_trace(str(root / f'{source.stem}-{mode}.trace.json'))
