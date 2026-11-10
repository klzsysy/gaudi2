"""Captured-input BF16/FP8 vendor SDPA comparison; not a model quality benchmark."""
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
import habana_frameworks.torch.utils.debug as debug
from habana_frameworks.torch.hpex.kernels import FusedSDPA
from habana_frameworks.torch.hpex.kernels.Fp8FusedSDPA import Fp8FusedSDPA
from moss_kv_candidate import attend, sdpa


torch.set_grad_enabled(False)
torch.set_num_threads(16)
root = Path(os.environ['MOSS_EXPERIMENT_DIR'])
cpu = torch.load(root / 'capture/kv-tree-8.pt', map_location='cpu', weights_only=True)
d = {name: value.to('hpu') if isinstance(value, torch.Tensor) else value for name, value in cpu.items()}
q, key, value, length, mask = (d[n] for n in ('q', 'key', 'value', 'length', 'mask'))
cache = SimpleNamespace(key=key, value=value, seqlen=length, index=d['index'], tracker={'tree': SimpleNamespace(mask=mask)})
reference = sdpa(cpu['q'].double(), cpu['key'].double(), cpu['value'].double(), cpu['length'], cpu['mask'])
baseline = attend(cache, q, d['k'], d['v'], 'compile-full').cpu().double()


def bias(q, key, length, mask):
    rows = q.shape[2]
    relative = torch.arange(key.shape[2], device=q.device) - length.view(-1, 1, 1, 1)
    visible = (relative < 0) | ((relative < rows) & ((mask.view(rows, 1) >> relative.clamp(0, rows - 1)) & 1).bool())
    return torch.zeros(visible.shape, dtype=torch.bfloat16, device=q.device).masked_fill_(~visible, float('-inf'))


def fused(q, key, value, length, mask, softmax):
    out = FusedSDPA.apply(q, key, value, bias(q, key, length, mask), 0.0, False, None, softmax, False)
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def quantize(x):
    scale = x.abs().amax().float().clamp_min(1e-8) / 240.0
    return torch.ops.hpu.cast_to_fp8_v2(x, scale.reciprocal(), False, False, torch.float8_e4m3fn)[0], scale


def fp8(q, key, value, length, mask, key_scale, value_scale, softmax_scale, softmax_inverse):
    packed_q, q_scale = quantize(q)
    out, _, _ = Fp8FusedSDPA.apply(packed_q, key, value, bias(q, key, length, mask), 0.0, False, None,
                                  'None', q_scale, key_scale, value_scale, softmax_scale, None, softmax_inverse)
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


packed_k, k_scale = quantize(key)
packed_v, v_scale = quantize(value)
p_scale = torch.full((), 240.0, device='hpu', dtype=torch.float32)
p_inverse = p_scale.reciprocal()
options = {'force_static_compile': True}
fused_compiled = torch.compile(fused, backend='hpu_backend', dynamic=None, options=options)
fp8_compiled = torch.compile(fp8, backend='hpu_backend', dynamic=None, options=options)
cases = [
    ('current-compiled', lambda: attend(cache, q, d['k'], d['v'], 'compile-full')),
    ('fused-bf16-default', lambda: fused_compiled(q, key, value, length, mask, 'None')),
    ('fused-bf16-fp32-softmax', lambda: fused_compiled(q, key, value, length, mask, 'fp32')),
    ('fused-fp8-resident-kv', lambda: fp8_compiled(q, packed_k, packed_v, length, mask, k_scale, v_scale, p_scale, p_inverse)),
]
results = []
for name, function in cases:
    try:
        actual = function()
        torch.hpu.synchronize()
        for _ in range(5):
            function()
        torch.hpu.synchronize()
        times = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(20):
                function()
            torch.hpu.synchronize()
            times.append((time.perf_counter() - start) * 1000 / 20)
        actual = actual.cpu().double()
        def error(a, b):
            return float((a-b).square().mean().sqrt() / b.square().mean().sqrt())
        record = {'name': name, 'median_ms': statistics.median(times), 'times_ms': times,
                  'finite': bool(actual.isfinite().all()), 'rrmse_original': error(actual, baseline),
                  'rrmse_fp64': error(actual, reference), 'fallback': debug._get_fallback_op_count()}
    except Exception as exc:
        record = {'name': name, 'error': type(exc).__name__ + ': ' + str(exc)[-2000:]}
    results.append(record)
    print(json.dumps(record), flush=True)
(root / 'fused-decode-results.json').write_text(json.dumps({'scope': 'One real first-layer verify input; KV quantization excluded, Q quantization included; no full model calibration or incremental KV update benchmark', 'results': results}, indent=2))
