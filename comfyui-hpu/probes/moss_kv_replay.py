"""Replay captured decode inputs across existing-operation candidates."""
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
import habana_frameworks.torch.utils.debug as debug
from moss_kv_candidate import attend, sdpa


torch.set_grad_enabled(False)
torch.set_num_threads(16)
root = Path(os.environ['MOSS_CAPTURE_DIR'])
results = []
for path in sorted(root.glob('kv-*.pt')):
    cpu = torch.load(path, map_location='cpu', weights_only=True)
    values = {k: v.to('hpu') if isinstance(v, torch.Tensor) else v for k, v in cpu.items()}
    cache = SimpleNamespace(key=values['key'], value=values['value'], index=values['index'], seqlen=values['length'],
                            tracker={'tree': None if values['mask'] is None else SimpleNamespace(mask=values['mask'])})
    q, k, v = [values[n] for n in ('q', 'k', 'v')]
    reference = sdpa(cpu['q'].double(), cpu['key'].double(), cpu['value'].double(), cpu['length'], cpu['mask'])
    original = attend(cache, q, k, v, 'baseline').cpu().double()
    for mode in os.environ.get('MOSS_KV_MODES', 'baseline,prefix,compile-full,compile-prefix,sdpa-prefix,sdpa-compile-prefix').split(','):
        start = time.perf_counter()
        result = attend(cache, q, k, v, mode)
        torch.hpu.synchronize()
        cold = time.perf_counter() - start
        for _ in range(4):
            attend(cache, q, k, v, mode)
        torch.hpu.synchronize()
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(20):
                attend(cache, q, k, v, mode)
            torch.hpu.synchronize()
            samples.append((time.perf_counter() - start) * 1000 / 20)
        actual = result.cpu().double()
        def rrmse(a, b):
            return ((a-b).square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-12)).item()
        record = {'capture': path.name, 'mode': mode, 'q': list(q.shape), 'kv': list(cache.key.shape),
                  'index': cache.index, 'cold_seconds': cold, 'median_ms': statistics.median(samples),
                  'ms': samples, 'finite': bool(actual.isfinite().all()), 'baseline_rrmse': rrmse(actual, original),
                  'fp64_rrmse': rrmse(actual, reference), 'baseline_fp64_rrmse': rrmse(original, reference),
                  'max_abs_diff': (actual-original).abs().max().item(), 'fallback': debug._get_fallback_op_count()}
        results.append(record)
        print(json.dumps(record), flush=True)
(root.parent / 'replay-results.json').write_text(json.dumps(results, indent=2))
