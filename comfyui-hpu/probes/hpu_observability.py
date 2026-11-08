"""Short, exclusive-card profiler smoke and effective eager calibration."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import habana_frameworks.torch.utils.debug as debug

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
torch.set_num_threads(16)
torch.manual_seed(42)


def measure(fn, repeats=5, iterations=20):
    for _ in range(3):
        fn()
    torch.hpu.synchronize()
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.hpu.synchronize()
        seconds.append((time.perf_counter() - start) / iterations)
    return {"seconds": seconds, "median_seconds": statistics.median(seconds),
            "iterations": iterations, "repeats": repeats}


x = torch.randn(256, 256, dtype=torch.bfloat16)
y = torch.randn(256, 256, dtype=torch.bfloat16)
a, b = x.to("hpu"), y.to("hpu")
c = a @ b
torch.testing.assert_close(c.cpu().float(), x.float() @ y.float(), atol=0.3, rtol=0.03)
for _ in range(3):
    c = torch.relu(a @ b)
torch.hpu.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                       torch.profiler.ProfilerActivity.HPU],
                            record_shapes=True) as prof:
    for _ in range(3):
        c = torch.relu(a @ b)
    torch.hpu.synchronize()
trace = args.output / "hpu-smoke.pt.trace.json"
prof.profiler.export_chrome_trace(str(trace))

result = {"torch": torch.__version__, "activities": [str(x) for x in torch.profiler.supported_activities()],
          "correctness": "PASS", "trace": str(trace), "mode": "eager"}
# Two 256 MiB buffers exceed Gaudi2's 48 MB SRAM. Copy counts read + write.
src = torch.ones(128 * 1024 * 1024, device="hpu", dtype=torch.bfloat16)
dst = torch.empty_like(src)
copy = measure(lambda: dst.copy_(src))
copy.update({"logical_bytes": 2 * src.numel() * src.element_size(),
             "effective_GB_s": 2 * src.numel() * src.element_size() / copy["median_seconds"] / 1e9})
result["copy_read_plus_write"] = copy
del src, dst
result["bf16_gemm"] = []
for m, n, k in [(4096, 4096, 4096), (1, 4096, 4096), (16, 4096, 4096), (1, 248320, 4096)]:
    a = torch.ones((m, k), device="hpu", dtype=torch.bfloat16)
    b = torch.ones((k, n), device="hpu", dtype=torch.bfloat16)
    c = torch.empty((m, n), device="hpu", dtype=torch.bfloat16)
    record = measure(lambda: torch.mm(a, b, out=c), iterations=10)
    record.update({"M": m, "N": n, "K": k, "logical_flops": 2*m*n*k,
                   "effective_TFLOP_s": 2*m*n*k / record["median_seconds"] / 1e12})
    result["bf16_gemm"].append(record)
    del a, b, c
a = torch.ones(16, device="hpu")
result["small_eager_add"] = measure(lambda: a.add_(1), iterations=100)
result["fallback_counts"] = debug._get_fallback_op_count()
result["memory_stats"] = torch.hpu.memory_stats()
(args.output / "calibration.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
