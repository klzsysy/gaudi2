"""Reproduce the public Gaudi2 FP8 shape benchmark with the local stack."""
import json
import os
import statistics
import time
from pathlib import Path

import habana_frameworks.torch.core as htcore
import torch

M, N, K = 6144, 11264, 5120
OUT = Path(os.environ.get("HPU_FP8_SHAPE_OUT", "fp8-shape.json"))
if os.environ.get("PT_HPU_LAZY_MODE") == "1":
    raise SystemExit("This probe intentionally refuses lazy mode: use a device-event benchmark with graph/output consumption.")
torch.set_grad_enabled(False)
torch.set_num_threads(16)


def fallback_counts():
    try:
        import habana_frameworks.torch.utils.debug as debug
        return dict(debug._get_fallback_op_count())
    except Exception:
        return {}


def run(dtype, iterations=20, repeats=5):
    a = torch.ones((M, K), device="hpu", dtype=dtype)
    b = torch.ones((K, N), device="hpu", dtype=dtype)
    for _ in range(10):
        torch.mm(a, b)
        if os.environ.get("PT_HPU_LAZY_MODE") == "1":
            htcore.mark_step()
    torch.hpu.synchronize()
    before = fallback_counts()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            torch.mm(a, b)
            if os.environ.get("PT_HPU_LAZY_MODE") == "1":
                htcore.mark_step()
        torch.hpu.synchronize()
        samples.append((time.perf_counter() - start) / iterations)
    after = fallback_counts()
    median = statistics.median(samples)
    flops = 2 * M * N * K
    return {"dtype": str(dtype), "shape_MNK": [M, N, K], "seconds": samples,
            "median_seconds": median, "flops": flops,
            "effective_TFLOP_s": flops / median / 1e12,
            "fallback_delta": {k: after.get(k, 0) - before.get(k, 0)
                               for k in set(before) | set(after)
                               if after.get(k, 0) != before.get(k, 0)},
            "output_dtype": str(torch.mm(a[:1], b).dtype)}


result = {"torch": torch.__version__, "lazy_mode": os.environ.get("PT_HPU_LAZY_MODE"),
          "device": torch.hpu.get_device_name(), "results": [run(torch.float8_e4m3fn)]}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
