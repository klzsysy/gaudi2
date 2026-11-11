"""Measure native FP8 with the imported MOSS weights and Gaudi2-compatible range."""
import json
import statistics
import time
from pathlib import Path
import torch
import habana_frameworks.torch.core
from safetensors import safe_open
from habana_frameworks.torch.utils.debug import _get_fallback_op_count
torch.set_grad_enabled(False)
torch.set_num_threads(32)
path = '/home/sonny/ComfyUI-Models/models/diffusion_models/Moody-Moss-Mix-V4.1_00001__clean-fp8.safetensors'
out = Path('/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-11-fp8')
print(torch.ops.hpu.fp8_gemm_v2._schemas, torch.ops.hpu.cast_to_fp8_v2._schemas, flush=True)

def native(x, w, sw, bias=None):
    flat = x.reshape(-1, x.shape[-1])
    scale = flat.abs().amax(-1, keepdim=True).float().clamp_min(1e-12) / 240.0
    quant = torch.ops.hpu.cast_to_fp8_v2(flat, scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
    y = torch.ops.hpu.fp8_gemm_v2(quant, False, w, True, None, x.dtype, scale, sw, bias, False)
    return y.reshape(*x.shape[:-1], w.shape[0])

def measure(fn, args):
    for _ in range(3):
        y = fn(*args)
    torch.hpu.synchronize()
    times = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(10):
            y = fn(*args)
        torch.hpu.synchronize()
        times.append((time.perf_counter() - t) / 10)
    return (y, statistics.median(times))
records = []
with safe_open(path, framework='pt', device='cpu') as f:
    for name in ('qkv', 'o', 'ffn_up_gate', 'ffn_down'):
        prefix = 'model.diffusion_model.layers.0.proj.' + name
        original = f.get_tensor(prefix + '.weight').float()
        scale = f.get_tensor(prefix + '.weight_scale').float()
        weight = (original / 2).to(torch.float8_e4m3fn)
        delta = weight.float() * 2 - original
        print(name, 'weight range', original.abs().max().item(), 'repack_rrmse', (delta.square().mean() / original.square().mean()).sqrt().item(), flush=True)
        w = weight.to('hpu')
        sw = (scale * 2).to('hpu')
        bf = (original * scale).bfloat16().to('hpu')
        for rows in (4119, 4403):
            torch.manual_seed(1)
            x = torch.randn((1, rows, bf.shape[1]), dtype=torch.bfloat16, device='hpu')
            ref, bt = measure(torch.compile(torch.nn.functional.linear, backend='hpu_backend', dynamic=False), (x, bf))
            y, ft = measure(torch.compile(native, backend='hpu_backend', dynamic=False), (x, w, sw))
            err = ((y.float() - ref.float()).square().mean() / ref.float().square().mean()).sqrt().item()
            r = dict(name=name, rows=rows, bf16_s=bt, fp8_s=ft, rrmse=err, finite=bool(y.isfinite().all().item()))
            records.append(r)
            print(r, flush=True)
(out / 'linear-probe.json').write_text(json.dumps(dict(records=records, fallback=_get_fallback_op_count()), indent=2))
