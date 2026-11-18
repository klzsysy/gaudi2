"""Benchmark complete end-to-end Plan A RMW Attention (including prep/tail/merge/transpose) vs D1."""
import argparse
import ctypes as C
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F
from habana_frameworks.torch.utils import _debug_eager_C
from habana_frameworks.torch.utils.experimental import _compute_stream
from comfy.cli_args import args as comfy_args
comfy_args.use_pytorch_cross_attention = True
from comfy.ldm.qwen_image21 import model as qwen_model
from qwen_d2_plan_a import address, build_library


def prep_split(q, k, v, split):
    qh = q.view(1, -1, 32, 128).transpose(1, 2).contiguous()
    kh = k.view(1, -1, 32, 128).transpose(1, 2)
    vh = v.view(1, -1, 32, 128).transpose(1, 2)
    scale = 128 ** -0.5
    kh = ((kh.float() - kh.float().mean(dim=-2, keepdim=True)) * scale).to(k.dtype)
    return (
        qh,
        kh[:, :, :split].contiguous(),
        vh[:, :, :split].contiguous(),
        kh[:, :, split:].contiguous(),
        vh[:, :, split:].contiguous(),
    )


def prep_split_and_tail_sdpa(q, k, v, split):
    qh = q.view(1, -1, 32, 128).transpose(1, 2).contiguous()
    kh = k.view(1, -1, 32, 128).transpose(1, 2)
    vh = v.view(1, -1, 32, 128).transpose(1, 2)
    scale = 128 ** -0.5
    kh = ((kh.float() - kh.float().mean(dim=-2, keepdim=True)) * scale).to(k.dtype)
    k1 = kh[:, :, :split].contiguous()
    v1 = vh[:, :, :split].contiguous()
    k2 = kh[:, :, split:].contiguous()
    v2 = vh[:, :, split:].contiguous()
    # qh is unscaled, k2 is already scaled by 128**-0.5, so scale=1.0
    last, max2, inv2 = torch.ops.hpu.sdpa_recomp_fwd(
        qh, k2, v2, None, 0.0, 1.0, False, True, 'fast', None, 'right')[:3]
    return qh, k1, v1, last, max2, inv2


def post_merge_rmw_tail(first, max1, inv1, last, max2, inv2):
    m1 = max1.float()
    m2 = max2.float()
    maximum = torch.maximum(m1, m2)
    # Both RMW main and RMW tail use raw softmax_fwd_hf8 (both scaled by 1/128, which cancels in share!)
    w1 = (m1 - maximum).exp() / inv1.float()
    w2 = (m2 - maximum).exp() / inv2.float()
    share = (w2 / (w1 + w2)).to(first.dtype)
    merged = torch.addcmul(first, last - first, share)
    return merged.transpose(1, 2).reshape(1, -1, 4096)


def post_merge_sdpa_tail(first, max1, inv1, last, max2, inv2):
    m1 = max1.float()
    m2 = max2.float()
    maximum = torch.maximum(m1, m2)
    # RMW main raw softmax_fwd_hf8 inv1 is 1/128 of SDPA's inv2; Check both 1.0 and 128.0 in validation
    w1 = (m1 - maximum).exp() / inv1.float()
    w2 = (m2 - maximum).exp() / inv2.float()
    share = (w2 / (w1 + w2)).to(first.dtype)
    merged = torch.addcmul(first, last - first, share)
    return merged.transpose(1, 2).reshape(1, -1, 4096)


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
    q_in = saved['q'].to('hpu').contiguous()
    k_in = saved['k'].to('hpu').contiguous()
    v_in = saved['v'].to('hpu').contiguous()
    queries = q_in.shape[1]
    full_keys = k_in.shape[1]
    aligned_keys = full_keys // 128 * 128
    tail_keys = full_keys - aligned_keys

    sample_rows = torch.linspace(0, queries - 1, 17).long()
    cq, ck, cv = [saved[name].reshape(1, -1, 32, 128).transpose(1, 2).double() for name in ('q', 'k', 'v')]
    ck_c = ck - ck.mean(dim=-2, keepdim=True)
    ref = (((cq[:, :, sample_rows] @ ck_c.transpose(-1, -2)) * (128 ** -0.5)).softmax(-1) @ cv).transpose(1, 2).reshape(1, 17, 4096)

    t_opts = {'hpu_fast_attention': True, 'block_index': 1}
    d1_fn = qwen_model._compiled_attention()
    for _ in range(4):
        d1_out = d1_fn(q_in, k_in, v_in, 32, None, t_opts, None)
    torch.hpu.synchronize()
    d1_cpu = d1_out.cpu()
    d1_rrmse = float((d1_cpu[:, sample_rows].double() - ref).norm() / ref.norm())
    d1_times = []
    for _ in range(7):
        t0 = time.perf_counter()
        for _ in range(5):
            d1_out = d1_fn(q_in, k_in, v_in, 32, None, t_opts, None)
        torch.hpu.synchronize()
        d1_times.append((time.perf_counter() - t0) / 5)

    compiled_prep_split = torch.compile(prep_split, backend='hpu_backend', dynamic=False, options={'force_static_compile': True})
    compiled_prep_sdpa_tail = torch.compile(prep_split_and_tail_sdpa, backend='hpu_backend', dynamic=False, options={'force_static_compile': True})
    compiled_post_rmw = torch.compile(post_merge_rmw_tail, backend='hpu_backend', dynamic=False, options={'force_static_compile': True})
    compiled_post_sdpa = torch.compile(post_merge_sdpa_tail, backend='hpu_backend', dynamic=False, options={'force_static_compile': True})

    results = {
        'input': str(options.input),
        'queries': queries,
        'full_keys': full_keys,
        'aligned_keys': aligned_keys,
        'tail_keys': tail_keys,
        'd1_full_baseline': {
            'median_ms': statistics.median(d1_times) * 1000,
            'samples_ms': [t * 1000 for t in d1_times],
            'fp64_rrmse': d1_rrmse,
        },
        'candidates': [],
    }
    print(f"D1 full attention: {results['d1_full_baseline']['median_ms']:.3f} ms | fp64_rrmse={d1_rrmse:.5f}", flush=True)

    # 1. Test Padded Mask with SRAM-staged mask (mask_mode = 1)
    padded_keys = (full_keys + 127) // 128 * 128
    pad_len = padded_keys - full_keys
    mask_bf16 = torch.zeros(padded_keys, dtype=torch.bfloat16, device='hpu')
    mask_bf16[full_keys:] = -float('inf')
    def prep_padded(q, k, v, plen):
        qh = q.view(1, -1, 32, 128).transpose(1, 2).contiguous()
        kh = k.view(1, -1, 32, 128).transpose(1, 2)
        vh = v.view(1, -1, 32, 128).transpose(1, 2)
        scale = 128 ** -0.5
        kh = ((kh.float() - kh.float().mean(dim=-2, keepdim=True)) * scale).to(k.dtype)
        return qh, F.pad(kh, (0, 0, 0, plen)).contiguous(), F.pad(vh, (0, 0, 0, plen)).contiguous()

    compiled_prep_padded = torch.compile(
        prep_padded,
        backend='hpu_backend', dynamic=False, options={'force_static_compile': True}
    )
    compiled_post_transpose = torch.compile(
        lambda out: out.transpose(1, 2).reshape(1, -1, 4096),
        backend='hpu_backend', dynamic=False, options={'force_static_compile': True}
    )
    mask_sram_bytes = (padded_keys * 2 + 255) & ~255
    for group_heads in (1, 2, 4, 8):
        tile = 480
        slots = 2
        while mask_sram_bytes + slots * ((padded_keys * tile * 2 + 255) & ~255) > 16 * 1024 * 1024:
            tile -= 8
        name = f"padded-sram-mask-g{group_heads}-s{slots}-t{tile}"
        t0 = time.perf_counter()
        plan = lib.build_plan_a(queries, padded_keys, group_heads, tile, slots, 1, 0, 1, 1, 1,
                                str(options.output / name).encode())
        compile_sec = time.perf_counter() - t0
        assert plan, f"Failed to build {name}"
        try:
            ws_bytes = lib.plan_a_workspace_bytes(plan)
            ws = torch.empty(max(256, ws_bytes), device='hpu', dtype=torch.uint8)
            ws_ptr = address(ws)
            aux_ptr = address(mask_bf16)

            def run_padded(q, k, v):
                qh, kh, vh = compiled_prep_padded(q, k, v, pad_len)
                out_bhnd = torch.empty_like(qh)
                _debug_eager_C.join_pending_pipeline_threads()
                ptrs = (C.c_uint64 * 6)(address(qh), address(kh), address(vh), address(out_bhnd), aux_ptr, 0)
                stream = _compute_stream()
                st = lib.launch_plan_a(plan, stream, ptrs, ws_ptr, 32)
                assert st == 0, st
                return compiled_post_transpose(out_bhnd)

            out1 = run_padded(q_in, k_in, v_in).cpu()
            out_alt = run_padded(q_in, k_in, v_in + 0.25).cpu()
            out2 = run_padded(q_in, k_in, v_in).cpu()
            assert out1.isfinite().all() and torch.equal(out1, out2) and not torch.equal(out1, out_alt)
            fp64_rrmse = float((out1[:, sample_rows].double() - ref).norm() / ref.norm())
            vs_d1_rrmse = float((out1.float() - d1_cpu.float()).norm() / d1_cpu.float().norm())
            for _ in range(3):
                _ = run_padded(q_in, k_in, v_in)
            torch.hpu.synchronize()
            samples = []
            for _ in range(7):
                t_start = time.perf_counter()
                for _ in range(5):
                    _ = run_padded(q_in, k_in, v_in)
                torch.hpu.synchronize()
                samples.append((time.perf_counter() - t_start) / 5)
            rec = dict(name=name, group_heads=group_heads, tile=tile, slots=slots,
                       compile_sec=compile_sec, workspace_bytes=ws_bytes,
                       fp64_rrmse=fp64_rrmse, vs_d1_rrmse=vs_d1_rrmse,
                       median_ms=statistics.median(samples) * 1000,
                       samples_ms=[s * 1000 for s in samples])
            results['candidates'].append(rec)
            (options.output / 'plan-a-split.json').write_text(json.dumps(results, indent=2))
            print(f"{name:32s}: {rec['median_ms']:7.3f} ms | fp64_rrmse={fp64_rrmse:.5f} | vs_d1={vs_d1_rrmse:.5f} | compile={compile_sec*1000:5.1f} ms", flush=True)
        finally:
            torch.hpu.synchronize()
            lib.destroy_plan_a(plan)

    for group_heads in (1, 2, 4):
        # Tile must be a multiple of 64 (128 bytes in bf16) when concatenating 1D retained stats!
        tile = 448
        slots = 2
        while slots * ((aligned_keys * tile * 2 + 255) & ~255) > 16 * 1024 * 1024:
            tile -= 64
        t0 = time.perf_counter()
        plan_main = lib.build_plan_a(queries, aligned_keys, group_heads, tile, slots, 1, 1, 1, 0, 1,
                                     str(options.output / f"main-g{group_heads}").encode())
        # For tail_keys (31 or 101), tile=8160 (1 tile per head!) fits in < 1.7 MiB of RMW!
        plan_tail = lib.build_plan_a(queries, tail_keys, group_heads, queries, 1, 1, 1, 1, 0, 1,
                                     str(options.output / f"tail-g{group_heads}").encode())
        compile_sec = time.perf_counter() - t0
        assert plan_main and plan_tail, f"Failed to build split plans for g{group_heads}"
        try:
            ws_bytes = max(lib.plan_a_workspace_bytes(plan_main), lib.plan_a_workspace_bytes(plan_tail))
            ws = torch.empty(max(256, ws_bytes), device='hpu', dtype=torch.uint8)
            ws_ptr = address(ws)
            stat_queries = (queries + 63) // 64 * 64

            # Mode A: RMW main + RMW tail
            def run_rmw_both(q, k, v):
                qh, k1, v1, k2, v2 = compiled_prep_split(q, k, v, aligned_keys)
                first = torch.empty_like(qh)
                last = torch.empty_like(qh)
                max1 = torch.empty(1, 32, stat_queries, 1, device='hpu', dtype=qh.dtype)
                inv1 = torch.empty_like(max1)
                max2 = torch.empty_like(max1)
                inv2 = torch.empty_like(max1)
                _debug_eager_C.join_pending_pipeline_threads()
                ptrs1 = (C.c_uint64 * 6)(address(qh), address(k1), address(v1), address(first), address(max1), address(inv1))
                ptrs2 = (C.c_uint64 * 6)(address(qh), address(k2), address(v2), address(last), address(max2), address(inv2))
                stream = _compute_stream()
                st1 = lib.launch_plan_a(plan_main, stream, ptrs1, ws_ptr, 32)
                st2 = lib.launch_plan_a(plan_tail, stream, ptrs2, ws_ptr, 32)
                assert st1 == 0 and st2 == 0, (st1, st2)
                return compiled_post_rmw(first, max1[:, :, :queries], inv1[:, :, :queries],
                                         last, max2[:, :, :queries], inv2[:, :, :queries])

            # Mode B: RMW main + fused SDPA tail in prep
            def run_sdpa_tail(q, k, v):
                qh, k1, v1, last, max2, inv2 = compiled_prep_sdpa_tail(q, k, v, aligned_keys)
                first = torch.empty_like(qh)
                max1 = torch.empty(1, 32, stat_queries, 1, device='hpu', dtype=qh.dtype)
                inv1 = torch.empty_like(max1)
                _debug_eager_C.join_pending_pipeline_threads()
                ptrs1 = (C.c_uint64 * 6)(address(qh), address(k1), address(v1), address(first), address(max1), address(inv1))
                stream = _compute_stream()
                st1 = lib.launch_plan_a(plan_main, stream, ptrs1, ws_ptr, 32)
                assert st1 == 0, st1
                return compiled_post_sdpa(first, max1[:, :, :queries], inv1[:, :, :queries], last, max2, inv2)

            for sub_name, fn in [(f"split-rmw-tail-g{group_heads}-s{slots}-t{tile}", run_rmw_both),
                                 (f"split-sdpa-tail-g{group_heads}-s{slots}-t{tile}", run_sdpa_tail)]:
                out1 = fn(q_in, k_in, v_in).cpu()
                out_alt = fn(q_in, k_in, v_in + 0.25).cpu()
                out2 = fn(q_in, k_in, v_in).cpu()
                assert out1.isfinite().all(), f"{sub_name} non-finite"
                assert torch.equal(out1, out2), f"{sub_name} A->B->A mismatch"
                assert not torch.equal(out1, out_alt), f"{sub_name} did not respond to v change"

                fp64_rrmse = float((out1[:, sample_rows].double() - ref).norm() / ref.norm())
                vs_d1_rrmse = float((out1.float() - d1_cpu.float()).norm() / d1_cpu.float().norm())

                for _ in range(3):
                    _ = fn(q_in, k_in, v_in)
                torch.hpu.synchronize()

                samples = []
                for _ in range(7):
                    t_start = time.perf_counter()
                    for _ in range(5):
                        _ = fn(q_in, k_in, v_in)
                    torch.hpu.synchronize()
                    samples.append((time.perf_counter() - t_start) / 5)

                rec = dict(name=sub_name, group_heads=group_heads, tile=tile, slots=slots,
                           compile_sec=compile_sec, workspace_bytes=ws_bytes,
                           fp64_rrmse=fp64_rrmse, vs_d1_rrmse=vs_d1_rrmse,
                           median_ms=statistics.median(samples) * 1000,
                           samples_ms=[s * 1000 for s in samples])
                results['candidates'].append(rec)
                (options.output / 'plan-a-split.json').write_text(json.dumps(results, indent=2))
                print(f"{sub_name:32s}: {rec['median_ms']:7.3f} ms | fp64_rrmse={fp64_rrmse:.5f} | vs_d1={vs_d1_rrmse:.5f} | compile={compile_sec*1000:5.1f} ms", flush=True)
        finally:
            torch.hpu.synchronize()
            lib.destroy_plan_a(plan_main)
            lib.destroy_plan_a(plan_tail)


if __name__ == '__main__':
    main()
