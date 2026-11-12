"""Check HPU RoPE decomposition against the existing opaque kitchen operation."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import comfy_kitchen
from comfy_kitchen.backends.eager.rope import apply_rope_split_half as eager_rope
from habana_frameworks.torch.utils.debug import _get_fallback_op_count
from habana_frameworks.torch.dynamo.compile_backend.decomposition import hpu_backend_decompositions_common
from moss_text_rope_candidate import install


parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('--mode', choices=['functional', 'foreach', 'opaque-none', 'opaque-functional-none'], default='functional')
options = parser.parse_args()
options.root.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
torch.set_num_threads(32)
records = []
operation = torch.ops.comfy_kitchen.apply_rope_split_half.default
assert operation not in hpu_backend_decompositions_common
opaque = options.mode.startswith('opaque')
if opaque:
    install(options.mode)


def measure(function, values):
    for _ in range(3):
        output = function(*values)
    torch.hpu.synchronize()
    samples = []
    for _ in range(5):
        start = time.perf_counter_ns()
        for _ in range(20):
            output = function(*values)
        torch.hpu.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e9 / 20)
    return tuple(x.cpu() for x in output), statistics.median(samples)


for dtype in (torch.bfloat16, torch.float32):
    for length in (1, 4, 8, 1970):
        for orthogonal in (False, True):
            torch.manual_seed(1200 + length)
            query = torch.randn(1, length, 16, 128, dtype=dtype).transpose(1, 2)[..., :64]
            key = torch.randn(1, length, 4, 128, dtype=dtype).transpose(1, 2)[..., :64]
            if orthogonal:
                phase = torch.randn(1, 1, length, 32)
                frequencies = torch.stack((phase.cos(), -phase.sin(), phase.sin(), phase.cos()), dim=-1).reshape(1, 1, length, 32, 2, 2)
            else:
                frequencies = torch.randn(1, 1, length, 32, 2, 2)
            values = tuple(x.to('hpu') for x in (query, key, frequencies))
            hpu_backend_decompositions_common.pop(operation, None)
            torch._dynamo.reset()
            baseline = eager_rope if opaque else torch.compile(
                comfy_kitchen.apply_rope_split_half, backend='hpu_backend', dynamic=False)
            expected, baseline_s = measure(baseline, values)
            if not opaque:
                install(options.mode)
            torch._dynamo.reset()
            compiled = torch.compile(comfy_kitchen.apply_rope_split_half, backend='hpu_backend', dynamic=False)
            actual, candidate_s = measure(compiled, values)
            differences = [a.float() - b.float() for a, b in zip(actual, expected)]
            record = dict(dtype=str(dtype), length=length, orthogonal=orthogonal,
                          equal=all(torch.equal(a, b) for a, b in zip(actual, expected)),
                          differing=sum(int((a != b).sum()) for a, b in zip(actual, expected)),
                          maximum=max(float(d.abs().max()) for d in differences),
                          baseline_s=baseline_s, candidate_s=candidate_s,
                          finite=all(bool(a.isfinite().all()) for a in actual), fallback=_get_fallback_op_count())
            records.append(record)
            (options.root / 'rope-lowering.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(record), flush=True)
            assert record['finite'] and not record['fallback'], record
