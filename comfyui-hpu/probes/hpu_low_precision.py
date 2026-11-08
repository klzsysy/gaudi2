"""Probe low precision capabilities and benchmark native paths when available.

This intentionally does not alter ComfyUI device policy. It distinguishes
native HPU compute from storage/dequantization and CPU fallback paths.
"""
import json
import statistics
import time
from pathlib import Path

import torch

OUT = Path(__import__("os").environ.get("HPU_LOW_PRECISION_OUT", "low-precision.json"))
torch.set_grad_enabled(False)
torch.set_num_threads(16)


def record_attempt(name, fn):
    try:
        value = fn()
        return {"name": name, "status": "PASS", "detail": str(value)}
    except Exception as exc:
        return {"name": name, "status": "FAIL", "error_type": type(exc).__name__,
                "error": str(exc)[:800]}


def benchmark(name, fn, flops, iterations=20, repeats=5):
    before_fallback = fallback_snapshot()
    for _ in range(3):
        fn()
    torch.hpu.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.hpu.synchronize()
        samples.append((time.perf_counter() - start) / iterations)
    median = statistics.median(samples)
    return {"name": name, "status": "PASS", "seconds": samples,
            "median_seconds": median, "flops": flops,
            "effective_TFLOP_s": flops / median / 1e12,
            "iterations": iterations, "repeats": repeats,
            "fallback_delta": fallback_delta(before_fallback, fallback_snapshot())}


def fallback_snapshot():
    try:
        import habana_frameworks.torch.utils.debug as debug
        return dict(debug._get_fallback_op_count())
    except Exception:
        return {}


def fallback_delta(before, after):
    keys = set(before) | set(after)
    return {key: after.get(key, 0) - before.get(key, 0) for key in keys
            if after.get(key, 0) != before.get(key, 0)}


result = {
    "torch": torch.__version__,
    "device": str(torch.hpu.get_device_name()),
    "supported_activities": [str(x) for x in torch.profiler.supported_activities()],
    "torch_float8_types": [x for x in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")
                           if hasattr(torch, x)],
    "hpu_ops": {name: hasattr(torch.ops.hpu, name) for name in (
        "fp8_gemm_v2", "fp8_gemm", "cast_to_fp8_v2", "convert_from_uint8")},
    "hpu_op_schemas": {},
    "capability_attempts": [], "benchmarks": [],
}
for name in ("fp8_gemm_v2", "fp8_gemm", "cast_to_fp8_v2"):
    if result["hpu_ops"][name]:
        try:
            result["hpu_op_schemas"][name] = str(getattr(torch.ops.hpu, name).default._schema)
        except Exception as exc:
            result["hpu_op_schemas"][name] = f"schema unavailable: {exc}"

for dtype_name in result["torch_float8_types"]:
    dtype = getattr(torch, dtype_name)
    result["capability_attempts"].append(record_attempt(
        f"allocate_{dtype_name}_on_hpu", lambda dtype=dtype: torch.ones((16, 16), device="hpu", dtype=dtype)))

result["capability_attempts"].append(record_attempt(
    "torch_mm_int8_on_hpu", lambda: torch.mm(torch.ones((16, 16), device="hpu", dtype=torch.int8),
                                               torch.ones((16, 16), device="hpu", dtype=torch.int8))))
result["capability_attempts"].append(record_attempt(
    "torch_int_mm_on_hpu", lambda: torch._int_mm(torch.ones((16, 32), device="hpu", dtype=torch.int8),
                                                   torch.ones((32, 16), device="hpu", dtype=torch.int8))))
try:
    import habana_frameworks.torch.hpex.experimental.transformer_engine as te
    result["transformer_engine"] = {"status": "imported", "module": te.__file__}
except Exception as exc:
    result["transformer_engine"] = {"status": "unavailable", "error": str(exc)[:800]}

# Compare like-for-like large MME shapes. FP8 is only attempted through native
# torch.mm if allocation and conversion are supported; a failed attempt is a
# capability result, not a substituted BF16 benchmark.
m = n = k = 4096
a_bf16 = torch.ones((m, k), device="hpu", dtype=torch.bfloat16)
b_bf16 = torch.ones((k, n), device="hpu", dtype=torch.bfloat16)
result["benchmarks"].append(benchmark("bf16_mm_4096", lambda: torch.mm(a_bf16, b_bf16), 2*m*n*k))

for dtype_name in result["torch_float8_types"]:
    dtype = getattr(torch, dtype_name)
    try:
        a = torch.ones((m, k), device="hpu", dtype=dtype)
        b = torch.ones((k, n), device="hpu", dtype=dtype)
        result["benchmarks"].append(benchmark(f"{dtype_name}_mm_4096", lambda: torch.mm(a, b), 2*m*n*k))
    except Exception as exc:
        result["benchmarks"].append({"name": f"{dtype_name}_mm_4096", "status": "FAIL",
                                      "error_type": type(exc).__name__, "error": str(exc)[:800]})

try:
    a = torch.ones((m, k), device="hpu", dtype=torch.int8)
    b = torch.ones((k, n), device="hpu", dtype=torch.int8)
    result["benchmarks"].append(benchmark("int8_torch_int_mm_4096",
                                          lambda: torch._int_mm(a, b), 2*m*n*k))
except Exception as exc:
    result["benchmarks"].append({"name": "int8_torch_int_mm_4096", "status": "FAIL",
                                  "error_type": type(exc).__name__, "error": str(exc)[:800]})

result["fallback_counts"] = {}
try:
    import habana_frameworks.torch.utils.debug as debug
    result["fallback_counts"] = debug._get_fallback_op_count()
except Exception as exc:
    result["fallback_error"] = str(exc)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
