"""Compile and benchmark Plan A RMW SRAM Attention variants on real Qwen inputs."""
import argparse
import ctypes as C
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch
from habana_frameworks.torch.utils import _debug_eager_C
from habana_frameworks.torch.utils.experimental import _compute_stream, _data_ptr
from comfy.cli_args import args as comfy_args
comfy_args.use_pytorch_cross_attention = True
from comfy.ldm.modules.attention import _hpu_fast_attention


def address(tensor):
    return _data_ptr(tensor) + tensor.storage_offset() * tensor.element_size()


def build_library(output_dir: Path) -> C.CDLL:
    src = Path(__file__).with_name('qwen_d2_plan_a.cpp')
    so = output_dir / 'libplan_a.so'
    subprocess.run([
        'g++', '-O2', '-std=gnu++17', '-fPIC', '-shared',
        '-I/usr/include/habanalabs', str(src),
        '-L/usr/lib/habanalabs', '-Wl,-rpath,/usr/lib/habanalabs',
        '-lSynapse', '-o', str(so)
    ], check=True)
    lib = C.CDLL(str(so))
    lib.build_plan_a.argtypes = [C.c_uint] * 5 + [C.c_int] * 5 + [C.c_char_p]
    lib.build_plan_a.restype = C.c_void_p
    lib.plan_a_workspace_bytes.argtypes = [C.c_void_p]
    lib.plan_a_workspace_bytes.restype = C.c_uint64
    lib.launch_plan_a.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(C.c_uint64), C.c_uint64, C.c_uint]
    lib.launch_plan_a.restype = C.c_int
    lib.destroy_plan_a.argtypes = [C.c_void_p]
    return lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(32)

    lib = build_library(options.output)
    saved = torch.load(options.input, weights_only=True)
    q_raw, k_raw, v_raw = [
        saved[name].reshape(1, -1, 32, 128).transpose(1, 2).contiguous().to('hpu')
        for name in ('q', 'k', 'v')
    ]
    queries = q_raw.shape[-2]
    full_keys = k_raw.shape[-2]
    aligned_keys = full_keys // 128 * 128

    # Pre-compute CPU FP64 reference on 17 sample rows across all 32 heads
    sample_rows = torch.linspace(0, queries - 1, 17).long()
    cq, ck, cv = [t.cpu().double() for t in (q_raw, k_raw, v_raw)]
    ck_centered = ck - ck.mean(dim=-2, keepdim=True)
    ref_full = ((cq[:, :, sample_rows] @ ck_centered.transpose(-1, -2)) * (128 ** -0.5)).softmax(-1) @ cv
    ref_aligned = ((cq[:, :, sample_rows] @ ck_centered[:, :, :aligned_keys].transpose(-1, -2)) * (128 ** -0.5)).softmax(-1) @ cv[:, :, :aligned_keys]

    # Measure D1 compiled baseline on the same input
    d1_compiled = torch.compile(_hpu_fast_attention, backend='hpu_backend', dynamic=None,
                                options={'force_static_compile': True})
    for _ in range(4):
        d1_out = d1_compiled(q_raw, k_raw, v_raw)
    torch.hpu.synchronize()
    d1_cpu = d1_out.cpu().double()[:, :, sample_rows]
    d1_rrmse = float((d1_cpu - ref_full).norm() / ref_full.norm())
    d1_times = []
    for _ in range(7):
        t0 = time.perf_counter()
        for _ in range(5):
            d1_out = d1_compiled(q_raw, k_raw, v_raw)
        torch.hpu.synchronize()
        d1_times.append((time.perf_counter() - t0) / 5)

    # Prepare scaled centered K and contiguous Q, V for RMW kernel calls
    scale = 128 ** -0.5
    k_scaled = ((k_raw.float() - k_raw.float().mean(dim=-2, keepdim=True)) * scale).to(k_raw.dtype).contiguous()
    q_cont = q_raw.contiguous()
    v_cont = v_raw.contiguous()

    # Also measure the cost of compiled K-centering + scaling alone
    def prep_k(k_in):
        return ((k_in.float() - k_in.float().mean(dim=-2, keepdim=True)) * (128 ** -0.5)).to(k_in.dtype).contiguous()
    compiled_prep_k = torch.compile(prep_k, backend='hpu_backend', dynamic=None, options={'force_static_compile': True})
    for _ in range(4):
        _ = compiled_prep_k(k_raw)
    torch.hpu.synchronize()
    prep_times = []
    for _ in range(7):
        t0 = time.perf_counter()
        for _ in range(10):
            _ = compiled_prep_k(k_raw)
        torch.hpu.synchronize()
        prep_times.append((time.perf_counter() - t0) / 10)

    results = {
        'input': str(options.input),
        'queries': queries,
        'full_keys': full_keys,
        'aligned_keys': aligned_keys,
        'd1_baseline': {
            'median_ms': statistics.median(d1_times) * 1000,
            'samples_ms': [t * 1000 for t in d1_times],
            'fp64_rrmse': d1_rrmse,
        },
        'prep_k_ms': statistics.median(prep_times) * 1000,
        'candidates': [],
    }
    print(f"D1 baseline: {results['d1_baseline']['median_ms']:.3f} ms (rrmse={d1_rrmse:.5f}), prep_k={results['prep_k_ms']:.3f} ms", flush=True)

    # Candidate configurations:
    # (name, keys, total_heads, group_heads, tile, slots, fast, retained, no_dma, kv_sram, use_split)
    configs = [
        # Single-head microbenchmarks (aligned_keys)
        ('h1-s1-t960-dma', aligned_keys, 1, 1, 960, 1, 1, 0, 0, 0, 0),
        ('h1-s1-t960-nodma', aligned_keys, 1, 1, 960, 1, 1, 0, 1, 0, 0),
        ('h1-s1-t960-nodma-split', aligned_keys, 1, 1, 960, 1, 1, 0, 1, 0, 1),
        ('h1-s2-t480-nodma-split', aligned_keys, 1, 1, 480, 2, 1, 0, 1, 0, 1),
        ('h1-s2-t448-nodma-split', aligned_keys, 1, 1, 448, 2, 1, 0, 1, 0, 1),
        ('h1-s1-t640-vsram', aligned_keys, 1, 1, 640, 1, 1, 0, 1, 1, 1),
        ('h1-s1-t640-kvsram', aligned_keys, 1, 1, 640, 1, 1, 0, 1, 2, 1),
        ('h1-s2-t320-kvsram', aligned_keys, 1, 1, 320, 2, 1, 0, 1, 2, 1),
        # 32-head full launches (aligned_keys, fast)
        ('h32-g1-s1-t960-nodma-split', aligned_keys, 32, 1, 960, 1, 1, 0, 1, 0, 1),
        ('h32-g1-s2-t480-nodma-split', aligned_keys, 32, 1, 480, 2, 1, 0, 1, 0, 1),
        ('h32-g2-s1-t960-nodma-split', aligned_keys, 32, 2, 960, 1, 1, 0, 1, 0, 1),
        ('h32-g2-s2-t480-nodma-split', aligned_keys, 32, 2, 480, 2, 1, 0, 1, 0, 1),
        ('h32-g4-s1-t960-nodma-split', aligned_keys, 32, 4, 960, 1, 1, 0, 1, 0, 1),
        ('h32-g4-s2-t480-nodma-split', aligned_keys, 32, 4, 480, 2, 1, 0, 1, 0, 1),
        ('h32-g1-s1-t640-vsram', aligned_keys, 32, 1, 640, 1, 1, 0, 1, 1, 1),
        ('h32-g1-s1-t640-kvsram', aligned_keys, 32, 1, 640, 1, 1, 0, 1, 2, 1),
        ('h32-g1-s2-t320-kvsram', aligned_keys, 32, 1, 320, 2, 1, 0, 1, 2, 1),
        # 32-head full unaligned keys (fast vs exact)
        ('h32-full-fast-g1-s1-t896', full_keys, 32, 1, 896, 1, 1, 0, 1, 0, 1),
        ('h32-full-fast-g1-s2-t448', full_keys, 32, 1, 448, 2, 1, 0, 1, 0, 1),
        ('h32-full-fast-g4-s2-t448', full_keys, 32, 4, 448, 2, 1, 0, 1, 0, 1),
        ('h32-full-exact-g1-s1-t896', full_keys, 32, 1, 896, 1, 0, 0, 1, 0, 1),
        ('h32-full-exact-g1-s2-t448', full_keys, 32, 1, 448, 2, 0, 0, 1, 0, 1),
        ('h32-full-exact-g4-s2-t448', full_keys, 32, 4, 448, 2, 0, 0, 1, 0, 1),
    ]

    _debug_eager_C.join_pending_pipeline_threads()
    stream = _compute_stream()
    assert stream

    for name, keys, total_heads, group_heads, tile, slots, fast, retained, no_dma, kv_sram, use_split in configs:
        # Adjust tile if 9445 exceeds 16MiB
        while ((kv_sram * ((keys * 128 * 2 + 255) & ~255)) + slots * ((keys * tile * 2 + 255) & ~255)) > 16 * 1024 * 1024:
            tile -= 32
        k_in = k_scaled[:, :total_heads, :keys].contiguous()
        v_in = v_cont[:, :total_heads, :keys].contiguous()
        q_in = q_cont[:, :total_heads].contiguous()
        out = torch.full_like(q_in, torch.nan)
        max_t = torch.empty(1, total_heads, queries, 1, device='hpu', dtype=q_in.dtype)
        inv_t = torch.empty_like(max_t)
        torch.hpu.synchronize()

        t0 = time.perf_counter()
        plan = lib.build_plan_a(queries, keys, group_heads, tile, slots, fast, retained,
                                no_dma, kv_sram, use_split, (str(options.output / name)).encode())
        compile_sec = time.perf_counter() - t0
        if not plan:
            print(f"SKIP {name}: build failed", flush=True)
            continue
        try:
            ws_bytes = lib.plan_a_workspace_bytes(plan)
            ws = torch.empty(max(256, ws_bytes), device='hpu', dtype=torch.uint8)
            ptrs = (C.c_uint64 * 6)(address(q_in), address(k_in), address(v_in),
                                    address(out), address(max_t), address(inv_t))
            ws_ptr = address(ws)

            status = lib.launch_plan_a(plan, stream, ptrs, ws_ptr, total_heads)
            assert status == 0, f"{name} launch status {status}"
            torch.hpu.synchronize()
            actual = out.cpu()
            assert actual.isfinite().all(), f"{name} non-finite output"

            ref = (ref_full if keys == full_keys else ref_aligned)[:, :total_heads]
            sample = actual[:, :, sample_rows].double()
            rrmse = float((sample - ref).norm() / ref.norm())

            for _ in range(3):
                lib.launch_plan_a(plan, stream, ptrs, ws_ptr, total_heads)
            torch.hpu.synchronize()

            samples = []
            for _ in range(7):
                t_start = time.perf_counter()
                for _ in range(5):
                    lib.launch_plan_a(plan, stream, ptrs, ws_ptr, total_heads)
                torch.hpu.synchronize()
                samples.append((time.perf_counter() - t_start) / 5)

            rec = dict(name=name, keys=keys, total_heads=total_heads, group_heads=group_heads,
                       tile=tile, slots=slots, fast=bool(fast), retained=bool(retained),
                       no_dma=bool(no_dma), kv_sram=kv_sram, use_split=bool(use_split),
                       compile_sec=compile_sec, workspace_bytes=ws_bytes,
                       fp64_rrmse=rrmse, median_ms=statistics.median(samples) * 1000,
                       samples_ms=[s * 1000 for s in samples])
            results['candidates'].append(rec)
            (options.output / 'plan-a-screen.json').write_text(json.dumps(results, indent=2))
            print(f"{name:30s}: {rec['median_ms']:7.3f} ms | compile={compile_sec*1000:6.1f} ms | ws={ws_bytes} | rrmse={rrmse:.5f}", flush=True)
        finally:
            torch.hpu.synchronize()
            lib.destroy_plan_a(plan)


if __name__ == '__main__':
    main()
