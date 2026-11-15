"""Known read/write workloads for calibrating Gaudi2 BMON counters."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from moss_native_profile import NativeProfile


parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("--operation", choices=["copy", "fill", "sum", "gemv", "idle"], default="copy")
parser.add_argument("--mib", type=int, default=512)
parser.add_argument("--iterations", type=int, default=20)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
torch.set_num_threads(32)
size = args.mib * 1024**2
source = torch.ones(size // 2, dtype=torch.bfloat16, device="hpu")
destination = torch.empty_like(source)
scalar = torch.empty((), dtype=torch.float32, device="hpu")
vector = torch.ones((1, 4096), dtype=torch.bfloat16, device="hpu")
weight = source.view(-1, 4096)
product = torch.empty((1, weight.shape[0]), dtype=torch.bfloat16, device="hpu")
operations = {
    "copy": lambda: destination.copy_(source),
    "fill": lambda: destination.fill_(1.25),
    "sum": lambda: torch.sum(source, dim=0, dtype=torch.float32, out=scalar),
    "gemv": lambda: torch.mm(vector, weight.T, out=product),
    "idle": lambda: time.sleep(0.0005),
}
sync = torch.hpu.synchronize
operation = operations[args.operation]
for _ in range(10):
    operation()
sync()
samples = []
for _ in range(5):
    start = time.perf_counter()
    for _ in range(args.iterations):
        operation()
    sync()
    samples.append(time.perf_counter() - start)
with NativeProfile(args.output):
    start = time.perf_counter()
    for _ in range(args.iterations):
        operation()
    sync()
    profiled = time.perf_counter() - start
read = size if args.operation in ("copy", "sum", "gemv") else 0
write = size if args.operation in ("copy", "fill") else 0
if args.operation == "gemv":
    read += vector.numel() * vector.element_size()
    write = product.numel() * product.element_size()
if args.operation == "sum":
    write = 4
result = dict(operation=args.operation, mib=args.mib, iterations=args.iterations,
              samples_seconds=samples, median_seconds=statistics.median(samples),
              profiled_seconds=profiled, known_read_bytes=read * args.iterations,
              known_write_bytes=write * args.iterations)
(args.output / "run.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
