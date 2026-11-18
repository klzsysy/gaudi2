"""Validate Plan A RMW SRAM Attention (split-sdpa-tail with q-scaling) against unit tests and 76 captured inputs."""
import argparse
import ctypes as C
import json
from pathlib import Path
import statistics
import time

import pytest
import torch
from torch._dynamo.backends.registry import lookup_backend
from habana_frameworks.torch.utils import _debug_eager_C
from habana_frameworks.torch.utils.experimental import _compute_stream
from comfy.cli_args import args as comfy_args
comfy_args.use_pytorch_cross_attention = True
from comfy.ldm.modules import attention as attn_mod
from comfy.ldm.qwen_image21 import model as qwen_model
from qwen_d2_plan_a import address, build_library


def _prep_aligned(q, k, v):
    scale = torch.full((), q.shape[-1] ** -0.5, dtype=q.dtype, device=q.device)
    qh = (q * scale).contiguous()
    kh = (k.float() - k.float().mean(dim=-2, keepdim=True)).to(k.dtype).contiguous()
    vh = v.contiguous()
    return qh, kh, vh


def _prep_split_and_tail(q, k, v, split):
    scale = torch.full((), q.shape[-1] ** -0.5, dtype=q.dtype, device=q.device)
    qh = (q * scale).contiguous()
    kh = (k.float() - k.float().mean(dim=-2, keepdim=True)).to(k.dtype)
    k1 = kh[..., :split, :].contiguous()
    v1 = v[..., :split, :].contiguous()
    k2 = kh[..., split:, :].contiguous()
    v2 = v[..., split:, :].contiguous()
    last, max2, inv2 = torch.ops.hpu.sdpa_recomp_fwd(
        qh, k2, v2, None, 0.0, 1.0, False, True, "fast", None, "right"
    )[:3]
    return qh, k1, v1, last, max2, inv2


def _merge_tail(first, max1, inv1, last, max2, inv2):
    m1 = max1.float()
    m2 = max2.float()
    maximum = torch.maximum(m1, m2)
    w1 = (m1 - maximum).exp() / inv1.float()
    w2 = (m2 - maximum).exp() / inv2.float()
    share = (w2 / (w1 + w2)).to(first.dtype)
    merged = torch.addcmul(first, last - first, share)
    return merged.transpose(1, 2).contiguous().transpose(1, 2)


def _transpose_nhd(out):
    return out.transpose(1, 2).contiguous().transpose(1, 2)


class PlanARuntime:
    def __init__(self, build_dir: Path, group_heads: int = 4, slots: int = 2):
        self.build_dir = build_dir
        self.lib = build_library(build_dir)
        self.group_heads = group_heads
        self.slots = slots
        self.plans = {}
        self.ws = None
        self.compiled_prep_aligned = torch.compile(
            _prep_aligned, backend="hpu_backend", dynamic=None, options={"force_static_compile": True}
        )
        self.compiled_prep_split = torch.compile(
            _prep_split_and_tail, backend="hpu_backend", dynamic=None, options={"force_static_compile": True}
        )
        self.compiled_merge = torch.compile(
            _merge_tail, backend="hpu_backend", dynamic=None, options={"force_static_compile": True}
        )
        self.compiled_transpose = torch.compile(
            _transpose_nhd, backend="hpu_backend", dynamic=None, options={"force_static_compile": True}
        )

    def get_plan(self, queries: int, aligned_keys: int, retained: int):
        key = (queries, aligned_keys, retained)
        plan = self.plans.get(key)
        if plan is not None:
            return plan
        # When retained=1, tile must be a multiple of 64 (128 bytes in bf16) for 1D stat concat alignment
        step = 64 if retained else 8
        max_tile = 448 if retained else 480
        tile = min(max_tile, (queries // step) * step or queries)
        while tile > 64 and self.slots * ((aligned_keys * tile * 2 + 255) & ~255) > 16 * 1024 * 1024:
            tile -= step
        name = f"rmw_q{queries}_k{aligned_keys}_r{retained}_g{self.group_heads}_s{self.slots}_t{tile}"
        plan = self.lib.build_plan_a(
            queries, aligned_keys, self.group_heads, tile, self.slots,
            1, retained, 1, 0, 1, str(self.build_dir / name).encode()
        )
        if not plan:
            raise RuntimeError(f"Failed to build Plan A recipe for {key}")
        self.plans[key] = plan
        return plan

    def get_workspace(self, ws_bytes: int, device):
        needed = max(256, int(ws_bytes))
        if self.ws is None or self.ws.numel() < needed or self.ws.device != device:
            self.ws = torch.empty(needed, dtype=torch.uint8, device=device)
        return self.ws

    def launch_aligned(self, qh: torch.Tensor, kh: torch.Tensor, vh: torch.Tensor) -> torch.Tensor:
        queries = qh.shape[-2]
        aligned_keys = kh.shape[-2]
        plan = self.get_plan(queries, aligned_keys, 0)
        ws_bytes = self.lib.plan_a_workspace_bytes(plan)
        ws = self.get_workspace(ws_bytes, qh.device)
        out = torch.empty_like(qh)
        _debug_eager_C.join_pending_pipeline_threads()
        ptrs = (C.c_uint64 * 6)(address(qh), address(kh), address(vh), address(out), 0, 0)
        st = self.lib.launch_plan_a(plan, _compute_stream(), ptrs, address(ws), 32)
        if st != 0:
            raise RuntimeError(f"launch_plan_a aligned failed with synStatus={st}")
        return self.compiled_transpose(out)

    def launch_split(self, qh: torch.Tensor, k1: torch.Tensor, v1: torch.Tensor,
                     last: torch.Tensor, max2: torch.Tensor, inv2: torch.Tensor) -> torch.Tensor:
        queries = qh.shape[-2]
        aligned_keys = k1.shape[-2]
        plan = self.get_plan(queries, aligned_keys, 1)
        ws_bytes = self.lib.plan_a_workspace_bytes(plan)
        ws = self.get_workspace(ws_bytes, qh.device)
        first = torch.empty_like(qh)
        stat_queries = (queries + 63) // 64 * 64
        max1 = torch.empty(1, 32, stat_queries, 1, device=qh.device, dtype=qh.dtype)
        inv1 = torch.empty_like(max1)
        _debug_eager_C.join_pending_pipeline_threads()
        ptrs = (C.c_uint64 * 6)(address(qh), address(k1), address(v1), address(first), address(max1), address(inv1))
        st = self.lib.launch_plan_a(plan, _compute_stream(), ptrs, address(ws), 32)
        if st != 0:
            raise RuntimeError(f"launch_plan_a split failed with synStatus={st}")
        return self.compiled_merge(first, max1[:, :, :queries], inv1[:, :, :queries], last, max2, inv2)

    @torch._dynamo.disable
    def execute(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        split = k.shape[-2] // 128 * 128
        if split == k.shape[-2]:
            qh, kh, vh = self.compiled_prep_aligned(q, k, v)
            return self.launch_aligned(qh, kh, vh)
        qh, k1, v1, last, max2, inv2 = self.compiled_prep_split(q, k, v, split)
        return self.launch_split(qh, k1, v1, last, max2, inv2)

    def close(self):
        if torch.hpu.is_available():
            torch.hpu.synchronize()
        for plan in self.plans.values():
            self.lib.destroy_plan_a(plan)
        self.plans.clear()
        self.ws = None


def make_hpu_fast_attention(runtime: PlanARuntime, fallback_fn):
    def _hpu_fast_attention_plan_a(q, k, v):
        if (q.device.type == "hpu"
                and q.dtype == k.dtype == v.dtype == torch.bfloat16
                and q.shape[:2] == (1, 32) and q.shape[-1] == 128 and q.shape[-2] >= 4096
                and k.shape == v.shape and k.shape[:2] == (1, 32)
                and k.shape[-1] == 128 and k.shape[-2] >= 4096):
            return runtime.execute(q, k, v)
        return fallback_fn(q, k, v)
    return _hpu_fast_attention_plan_a


def framework_check(matrix, other, vector):
    scale = torch.full((), .125, dtype=torch.float32, device=matrix.device)
    left = torch.ops.hpu.cast_to_fp8_v2(matrix, 8.0, False, False, torch.float8_e4m3fn)[0]
    right = torch.ops.hpu.cast_to_fp8_v2(other, 8.0, False, False, torch.float8_e4m3fn)[0]
    product = torch.ops.hpu.fp8_gemm_v2(left, False, right, False, None, torch.bfloat16,
                                      scale, scale, None, False)
    return (matrix @ other, product, vector.exp(), vector.sigmoid(), vector.tanh(),
            vector.rsqrt(), vector.log(), vector.sin(), vector.cos(), vector.softmax(-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', type=Path)
    parser.add_argument('output', type=Path)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(32)

    orig_fast_attn = attn_mod._hpu_fast_attention
    runtime = PlanARuntime(options.output, group_heads=4, slots=2)
    try:
        attn_mod._hpu_fast_attention = make_hpu_fast_attention(runtime, orig_fast_attn)
        qwen_model._compiled_attention.cache_clear()
        torch._dynamo.reset()

        # Diagnostic on graph count for KV boundary test
        backend = lookup_backend('hpu_backend')
        diag_graphs = []
        def counted_backend(graph, inputs, **kwargs):
            diag_graphs.append(graph.code)
            return backend(graph, inputs, **kwargs)
        compiled_diag = torch.compile(qwen_model._attention, backend=counted_backend, dynamic=None,
                                      options={'force_static_compile': True})
        lengths = (8191, 8192, 8193, 8256, 8257, 8319, 8320, 8321, 9445, 9400, 8191)
        query_diag = torch.zeros(1, 8160, 4096, device='hpu', dtype=torch.bfloat16)
        for idx, length in enumerate(lengths):
            val = .25 if idx == len(lengths) - 1 else .25 + idx / 8
            key_diag = torch.full((1, length, 4096), .25, device='hpu', dtype=torch.bfloat16)
            vals_diag = torch.full_like(key_diag, val)
            _ = compiled_diag(query_diag, key_diag, vals_diag, 32, None, {'hpu_fast_attention': True, 'block_index': 1}, None)
            print(f"After idx={idx} len={length}: diag_graphs={len(diag_graphs)}", flush=True)
        for gi, gcode in enumerate(diag_graphs):
            print(f"--- DIAG GRAPH {gi} ---\n{gcode}", flush=True)

        # Diagnostic on Q=4096, K=4225 (seed=413)
        gen413 = torch.Generator().manual_seed(413)
        q413 = torch.randn(1, 4096, 4096, generator=gen413, dtype=torch.bfloat16).to('hpu')
        k413 = torch.randn(1, 4225, 4096, generator=gen413, dtype=torch.bfloat16).to('hpu')
        v413 = torch.randn(1, 4225, 4096, generator=gen413, dtype=torch.bfloat16).to('hpu')
        qh413 = q413.reshape(1, 4096, 32, 128).transpose(1, 2)
        kh413 = k413.reshape(1, 4225, 32, 128).transpose(1, 2)
        vh413 = v413.reshape(1, 4225, 32, 128).transpose(1, 2)
        kc413 = (kh413.float() - kh413.float().mean(dim=-2, keepdim=True)).to(kh413.dtype)
        d1_f4224, d1_m4224, d1_i4224 = torch.ops.hpu.sdpa_recomp_fwd(
            qh413.contiguous(), kc413[:, :, :4224].contiguous(), vh413[:, :, :4224].contiguous(),
            None, 0.0, 128 ** -0.5, False, True, "fast", None, "right"
        )[:3]
        for q_mode, q_scaled in [
            ("bf16_scale", (qh413.contiguous() * (128 ** -0.5)).contiguous()),
            ("fp32_scale", (qh413.float() * (128 ** -0.5)).to(qh413.dtype).contiguous()),
        ]:
            pa_f4224 = runtime.launch_aligned(q_scaled, kc413[:, :, :4224].contiguous(), vh413[:, :, :4224].contiguous())
            diff4224 = float((pa_f4224.double() - d1_f4224.double()).norm() / d1_f4224.double().norm())
            print(f"4224 {q_mode} vs d1_f4224 rrmse = {diff4224:.6f}", flush=True)

        print("=== Running pytest hpu_fast_attention_test.py ===", flush=True)
        ret = pytest.main(["-v", "tests-unit/comfy_test/hpu_fast_attention_test.py"])
        assert ret == 0, f"pytest hpu_fast_attention_test.py failed with {ret}"

        # 2. Validate across all 76 captured image inputs + framework checks
        qwen_model._compiled_attention.cache_clear()
        torch._dynamo.reset()

        d1_compiled = torch.compile(
            lambda q, k, v, t_opts: orig_fast_attn(
                q.view(1, -1, 32, 128).transpose(1, 2),
                k.view(1, -1, 32, 128).transpose(1, 2),
                v.view(1, -1, 32, 128).transpose(1, 2),
            ).transpose(1, 2).reshape(1, -1, 4096),
            backend='hpu_backend', dynamic=None, options={'force_static_compile': True}
        )
        plan_a_compiled = qwen_model._compiled_attention()

        generator = torch.Generator().manual_seed(814)
        checks = (torch.randn(128, 128, generator=generator).bfloat16().to('hpu'),
                  torch.randn(128, 256, generator=generator).bfloat16().to('hpu'),
                  (torch.rand(64, 128, generator=generator) * 3 + .25).to('hpu'))
        framework = torch.compile(framework_check, backend='hpu_backend', dynamic=False)
        expected_checks = [value.cpu() for value in framework(*checks)]

        results = []
        paths = sorted(options.inputs.glob('*.pt'))
        t_opts = {'hpu_fast_attention': True, 'block_index': 1}
        for path in paths:
            if 'masked' in path.name:
                continue
            saved = torch.load(path, weights_only=True)
            q_cpu, k_cpu, v_cpu = saved['q'], saved['k'], saved['v']
            q = q_cpu.to('hpu').contiguous()
            k = k_cpu.to('hpu').contiguous()
            v = v_cpu.to('hpu').contiguous()

            actual = plan_a_compiled(q, k, v, 32, None, t_opts, None).cpu()
            assert actual.isfinite().all(), f"Non-finite on {path.name}"

            rows = torch.linspace(0, q.shape[1] - 1, 17).long()
            cq, ck, cv = [t.reshape(1, -1, 32, 128).transpose(1, 2).double() for t in (q_cpu, k_cpu, v_cpu)]
            ck_c = ck - ck.mean(dim=-2, keepdim=True)
            reference = (((cq[:, :, rows] @ ck_c.transpose(-1, -2)) / 128 ** .5).softmax(-1) @ cv).transpose(1, 2).reshape(1, 17, 4096)
            sample = actual[:, rows].double()
            fp64_rrmse = float((sample - reference).norm() / reference.norm())

            d1_out = d1_compiled(q, k, v, t_opts).cpu()
            d1_rrmse = float((d1_out[:, rows].double() - reference).norm() / reference.norm())
            vs_d1_rrmse = float((actual.float() - d1_out.float()).norm() / d1_out.float().norm())
            bitwise_d1 = bool(torch.equal(actual, d1_out))

            record = dict(
                input=path.name,
                queries=int(q.shape[1]),
                keys=int(k.shape[1]),
                fp64_rrmse=fp64_rrmse,
                d1_fp64_rrmse=d1_rrmse,
                vs_d1_rrmse=vs_d1_rrmse,
                bitwise_d1=bitwise_d1,
            )

            if path.name in ('forward-07-layer-15-part-1.pt', 'forward-08-layer-15-part-1.pt'):
                first = actual.clone()
                altered = plan_a_compiled(q, k, v + .25, 32, None, t_opts, None).cpu()
                returned = plan_a_compiled(q, k, v, 32, None, t_opts, None).cpu()
                assert torch.equal(first, returned)
                assert not torch.equal(first, altered)
                query_changed = plan_a_compiled(q.roll(32, dims=1), k, v, 32, None, t_opts, None).cpu()
                key_changed = plan_a_compiled(q, k.roll(64, dims=1), v, 32, None, t_opts, None).cpu()
                assert not torch.equal(first, query_changed)
                assert not torch.equal(first, key_changed)
                assert torch.equal(first, plan_a_compiled(q, k, v, 32, None, t_opts, None).cpu())
                assert torch.equal(q.cpu(), q_cpu) and torch.equal(k.cpu(), k_cpu) and torch.equal(v.cpu(), v_cpu)

                for _ in range(3):
                    d1_compiled(q, k, v, t_opts)
                    plan_a_compiled(q, k, v, 32, None, t_opts, None)
                torch.hpu.synchronize()

                times = {'d1': [], 'plan_a': []}
                for index in range(7):
                    order = [
                        ('d1', lambda: d1_compiled(q, k, v, t_opts)),
                        ('plan_a', lambda: plan_a_compiled(q, k, v, 32, None, t_opts, None)),
                    ]
                    if index % 2:
                        order.reverse()
                    for name, fn in order:
                        torch.hpu.synchronize()
                        started = time.perf_counter()
                        for _ in range(5):
                            fn()
                        torch.hpu.synchronize()
                        times[name].append((time.perf_counter() - started) / 5)
                record.update(
                    value_change=True, query_change=True, key_change=True,
                    return_equal=True, inputs_unchanged=True,
                    median_ms={name: statistics.median(vals) * 1000 for name, vals in times.items()},
                    samples_ms={name: [v * 1000 for v in vals] for name, vals in times.items()},
                )
            results.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != 'samples_ms'}), flush=True)

        check_equal = [torch.equal(value.cpu(), expected) for value, expected in zip(framework(*checks), expected_checks)]
        assert all(check_equal), check_equal
        assert len(runtime.plans) >= 2, f"Expected compiled RMW plans, got {len(runtime.plans)}"

        summary = {
            'total_validated': len(results),
            'max_fp64_rrmse': max(r['fp64_rrmse'] for r in results),
            'median_fp64_rrmse': statistics.median(r['fp64_rrmse'] for r in results),
            'max_d1_fp64_rrmse': max(r['d1_fp64_rrmse'] for r in results),
            'median_d1_fp64_rrmse': statistics.median(r['d1_fp64_rrmse'] for r in results),
            'max_vs_d1_rrmse': max(r['vs_d1_rrmse'] for r in results),
            'median_vs_d1_rrmse': statistics.median(r['vs_d1_rrmse'] for r in results),
            'bitwise_d1_count': sum(1 for r in results if r['bitwise_d1']),
            'compiled_plan_count': len(runtime.plans),
            'framework_checks_passed': all(check_equal),
            'results': results,
        }
        (options.output / 'validation-76.json').write_text(json.dumps(summary, indent=2))
        print("=== SUMMARY ===", json.dumps({k: v for k, v in summary.items() if k != 'results'}, indent=2), flush=True)
    finally:
        attn_mod._hpu_fast_attention = orig_fast_attn
        runtime.close()


if __name__ == '__main__':
    main()
