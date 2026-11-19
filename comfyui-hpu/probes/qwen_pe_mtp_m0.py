"""M0: Small-M FP8 bandwidth microbenchmark, MTP attention microbenchmark, and FP8 MTP baseline screen."""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
import habana_frameworks.torch
from habana_frameworks.torch.hpex.kernels import FusedSDPA
from safetensors import safe_open

from comfy.cli_args import args as comfy_args
comfy_args.disable_dynamic_vram = True
comfy_args.disable_pinned_memory = True
comfy_args.disable_async_offload = True
comfy_args.use_pytorch_cross_attention = True

import comfy.sd
from comfy.text_encoders import hpu_kv_attention, qwen35


def _quantize_fp8(x, divisor):
    flat = x.reshape(-1, x.shape[-1])
    scale = flat.abs().amax(-1, keepdim=True).float().clamp_min(1e-12) / divisor
    q = torch.ops.hpu.cast_to_fp8_v2(flat, scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
    return q, scale


def _bench(fn, warmup=6, iters=30, repeats=5):
    for _ in range(warmup):
        fn()
    torch.hpu.synchronize()
    times = []
    for _ in range(repeats):
        torch.hpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.hpu.synchronize()
        times.append((time.perf_counter() - t0) / iters)
    return statistics.median(times)


def run_matrix_micro(model_path: Path, out_dir: Path):
    print("=== M0.A: Small-M Rotated FP8 Linear Sustainable Bandwidth ===", flush=True)
    divisor = torch.tensor(240.0, dtype=torch.float32, device="hpu")

    # Load 4 layers of real weights (>48MB SRAM easily) for each group + lm_head
    with safe_open(str(model_path), framework="pt", device="cpu") as f:
        def load_pair(key):
            w = f.get_tensor(key).to("hpu")
            s = f.get_tensor(key.removesuffix(".weight") + ".weight_scale").to("hpu")
            return w, s

        mlp_layers = []
        for idx in range(4):
            p = f"model.language_model.layers.{idx}.mlp."
            wg, sg = load_pair(p + "gate_proj.weight")
            wu, su = load_pair(p + "up_proj.weight")
            wd, sd = load_pair(p + "down_proj.weight")
            wgu = torch.cat([wg, wu], dim=0).contiguous()
            sgu = torch.cat([sg.reshape(1).expand(wg.shape[0]), su.reshape(1).expand(wu.shape[0])]).reshape(1, -1).contiguous()
            mlp_layers.append((wg, sg, wu, su, wgu, sgu, wd, sd))

        gdn_layers = []
        for idx in [0, 1, 2, 4, 5, 6]:
            p = f"model.language_model.layers.{idx}.linear_attn."
            wqkv, sqkv = load_pair(p + "in_proj_qkv.weight")
            wz, sz = load_pair(p + "in_proj_z.weight")
            wo, so = load_pair(p + "out_proj.weight")
            wqkvz = torch.cat([wqkv, wz], dim=0).contiguous()
            sqkvz = torch.cat([sqkv.reshape(1).expand(wqkv.shape[0]), sz.reshape(1).expand(wz.shape[0])]).reshape(1, -1).contiguous()
            gdn_layers.append((wqkv, sqkv, wz, sz, wqkvz, sqkvz, wo, so))

        attn_layers = []
        for idx in [3, 7, 11, 15, 19, 23]:
            p = f"model.language_model.layers.{idx}.self_attn."
            wq, sq = load_pair(p + "q_proj.weight")
            wk, sk = load_pair(p + "k_proj.weight")
            wv, sv = load_pair(p + "v_proj.weight")
            wo, so = load_pair(p + "o_proj.weight")
            wqkv = torch.cat([wq, wk, wv], dim=0).contiguous()
            sqkv = torch.cat([
                sq.reshape(1).expand(wq.shape[0]),
                sk.reshape(1).expand(wk.shape[0]),
                sv.reshape(1).expand(wv.shape[0]),
            ]).reshape(1, -1).contiguous()
            attn_layers.append((wq, sq, wk, sk, wv, sv, wqkv, sqkv, wo, so))

        lm_w, lm_s = load_pair("lm_head.weight")

    def mlp_sep(xs):
        outs = []
        for x, (wg, sg, wu, su, _, _, wd, sd) in zip(xs, mlp_layers):
            qg, s1 = _quantize_fp8(x, divisor)
            g = torch.ops.hpu.fp8_gemm_v2(qg, False, wg, True, None, x.dtype, s1, sg, None, False)
            qu, s2 = _quantize_fp8(x, divisor)
            u = torch.ops.hpu.fp8_gemm_v2(qu, False, wu, True, None, x.dtype, s2, su, None, False)
            h = torch.nn.functional.silu(g) * u
            qh, s3 = _quantize_fp8(h, divisor)
            outs.append(torch.ops.hpu.fp8_gemm_v2(qh, False, wd, True, None, x.dtype, s3, sd, None, False))
        return outs

    def mlp_shared(xs):
        outs = []
        for x, (wg, sg, wu, su, _, _, wd, sd) in zip(xs, mlp_layers):
            q, s1 = _quantize_fp8(x, divisor)
            g = torch.ops.hpu.fp8_gemm_v2(q, False, wg, True, None, x.dtype, s1, sg, None, False)
            u = torch.ops.hpu.fp8_gemm_v2(q, False, wu, True, None, x.dtype, s1, su, None, False)
            h = torch.nn.functional.silu(g) * u
            qh, s3 = _quantize_fp8(h, divisor)
            outs.append(torch.ops.hpu.fp8_gemm_v2(qh, False, wd, True, None, x.dtype, s3, sd, None, False))
        return outs

    def mlp_merged(xs):
        outs = []
        for x, (_, _, _, _, wgu, sgu, wd, sd) in zip(xs, mlp_layers):
            q, s1 = _quantize_fp8(x, divisor)
            gu = torch.ops.hpu.fp8_gemm_v2(q, False, wgu, True, None, x.dtype, s1, sgu, None, False)
            g, u = gu.chunk(2, dim=-1)
            h = torch.nn.functional.silu(g) * u
            qh, s3 = _quantize_fp8(h, divisor)
            outs.append(torch.ops.hpu.fp8_gemm_v2(qh, False, wd, True, None, x.dtype, s3, sd, None, False))
        return outs

    def gdn_sep(xs):
        outs = []
        for x, (wqkv, sqkv, wz, sz, _, _, _, _) in zip(xs, gdn_layers):
            q1, s1 = _quantize_fp8(x, divisor)
            a = torch.ops.hpu.fp8_gemm_v2(q1, False, wqkv, True, None, x.dtype, s1, sqkv, None, False)
            q2, s2 = _quantize_fp8(x, divisor)
            b = torch.ops.hpu.fp8_gemm_v2(q2, False, wz, True, None, x.dtype, s2, sz, None, False)
            outs.append((a, b))
        return outs

    def gdn_merged(xs):
        outs = []
        for x, (_, _, _, _, wqkvz, sqkvz, _, _) in zip(xs, gdn_layers):
            q, s1 = _quantize_fp8(x, divisor)
            ab = torch.ops.hpu.fp8_gemm_v2(q, False, wqkvz, True, None, x.dtype, s1, sqkvz, None, False)
            outs.append(ab.split((8192, 4096), dim=-1))
        return outs

    def attn_sep(xs):
        outs = []
        for x, (wq, sq, wk, sk, wv, sv, _, _, _, _) in zip(xs, attn_layers):
            q1, s1 = _quantize_fp8(x, divisor)
            a = torch.ops.hpu.fp8_gemm_v2(q1, False, wq, True, None, x.dtype, s1, sq, None, False)
            q2, s2 = _quantize_fp8(x, divisor)
            b = torch.ops.hpu.fp8_gemm_v2(q2, False, wk, True, None, x.dtype, s2, sk, None, False)
            q3, s3 = _quantize_fp8(x, divisor)
            c = torch.ops.hpu.fp8_gemm_v2(q3, False, wv, True, None, x.dtype, s3, sv, None, False)
            outs.append((a, b, c))
        return outs

    def attn_merged(xs):
        outs = []
        for x, (_, _, _, _, _, _, wqkv, sqkv, _, _) in zip(xs, attn_layers):
            q, s1 = _quantize_fp8(x, divisor)
            abc = torch.ops.hpu.fp8_gemm_v2(q, False, wqkv, True, None, x.dtype, s1, sqkv, None, False)
            outs.append(abc.split((8192, 1024, 1024), dim=-1))
        return outs

    def lm_head_fn(x):
        q, s1 = _quantize_fp8(x, divisor)
        return torch.ops.hpu.fp8_gemm_v2(q, False, lm_w, True, None, x.dtype, s1, lm_s, None, False)

    c_mlp_sep = torch.compile(mlp_sep, backend="hpu_backend", fullgraph=True, dynamic=False, options={"force_static_compile": True})
    c_mlp_shared = torch.compile(mlp_shared, backend="hpu_backend", fullgraph=True, dynamic=False, options={"force_static_compile": True})
    c_mlp_merged = torch.compile(mlp_merged, backend="hpu_backend", fullgraph=True, dynamic=False, options={"force_static_compile": True})
    c_gdn_sep = torch.compile(gdn_sep, backend="hpu_backend", fullgraph=True, dynamic=False, options={"force_static_compile": True})
    c_gdn_merged = torch.compile(gdn_merged, backend="hpu_backend", fullgraph=True, dynamic=False, options={"force_static_compile": True})
    c_attn_sep = torch.compile(attn_sep, backend="hpu_backend", fullgraph=True, dynamic=False, options={"force_static_compile": True})
    c_attn_merged = torch.compile(attn_merged, backend="hpu_backend", fullgraph=True, dynamic=False, options={"force_static_compile": True})
    c_lm = torch.compile(lm_head_fn, backend="hpu_backend", fullgraph=True, dynamic=False, options={"force_static_compile": True})

    mlp_bytes = sum(wg.numel() + wu.numel() + wd.numel() for wg, _, wu, _, _, _, wd, _ in mlp_layers)
    gdn_bytes = sum(wqkv.numel() + wz.numel() for wqkv, _, wz, _, _, _, _, _ in gdn_layers)
    attn_bytes = sum(wq.numel() + wk.numel() + wv.numel() for wq, _, wk, _, wv, _, _, _, _, _ in attn_layers)
    lm_bytes = lm_w.numel()

    rows = []
    torch.manual_seed(7)
    for m in (1, 2, 4, 8, 16):
        xs_mlp = [torch.randn(m, 4096, device="hpu", dtype=torch.bfloat16) for _ in mlp_layers]
        ref_mlp = [t.cpu().float() for t in c_mlp_sep(xs_mlp)]
        for name, fn in [("mlp_sep", c_mlp_sep), ("mlp_shared", c_mlp_shared), ("mlp_merged", c_mlp_merged)]:
            out = [t.cpu().float() for t in fn(xs_mlp)]
            rrmse = max(((a - b).square().mean() / b.square().mean().clamp_min(1e-12)).sqrt().item() for a, b in zip(out, ref_mlp))
            sec = _bench(lambda: fn(xs_mlp), iters=25)
            tbps = mlp_bytes / sec / 1e12
            rec = dict(group="mlp_4L", variant=name, M=m, ms=round(sec * 1e3, 4),
                       weight_MB=round(mlp_bytes / 1e6, 2), TBps=round(tbps, 4),
                       pct_peak=round(tbps / 2.45 * 100, 2), rrmse=round(rrmse, 6))
            rows.append(rec)
            print(" ", rec, flush=True)

        xs_gdn = [torch.randn(m, 4096, device="hpu", dtype=torch.bfloat16) for _ in gdn_layers]
        ref_gdn = [(a.cpu().float(), b.cpu().float()) for a, b in c_gdn_sep(xs_gdn)]
        for name, fn in [("gdn_qkvz_sep", c_gdn_sep), ("gdn_qkvz_merged", c_gdn_merged)]:
            out = [(a.cpu().float(), b.cpu().float()) for a, b in fn(xs_gdn)]
            rrmse = max(
                max(((a1 - a2).square().mean() / a2.square().mean().clamp_min(1e-12)).sqrt().item(),
                    ((b1 - b2).square().mean() / b2.square().mean().clamp_min(1e-12)).sqrt().item())
                for (a1, b1), (a2, b2) in zip(out, ref_gdn)
            )
            sec = _bench(lambda: fn(xs_gdn), iters=25)
            tbps = gdn_bytes / sec / 1e12
            rec = dict(group="gdn_6L_in", variant=name, M=m, ms=round(sec * 1e3, 4),
                       weight_MB=round(gdn_bytes / 1e6, 2), TBps=round(tbps, 4),
                       pct_peak=round(tbps / 2.45 * 100, 2), rrmse=round(rrmse, 6))
            rows.append(rec)
            print(" ", rec, flush=True)

        xs_attn = [torch.randn(m, 4096, device="hpu", dtype=torch.bfloat16) for _ in attn_layers]
        ref_attn = [(a.cpu().float(), b.cpu().float(), c.cpu().float()) for a, b, c in c_attn_sep(xs_attn)]
        for name, fn in [("attn_qkv_sep", c_attn_sep), ("attn_qkv_merged", c_attn_merged)]:
            out = [(a.cpu().float(), b.cpu().float(), c.cpu().float()) for a, b, c in fn(xs_attn)]
            rrmse = max(
                max(((x1 - x2).square().mean() / x2.square().mean().clamp_min(1e-12)).sqrt().item() for x1, x2 in zip(p1, p2))
                for p1, p2 in zip(out, ref_attn)
            )
            sec = _bench(lambda: fn(xs_attn), iters=25)
            tbps = attn_bytes / sec / 1e12
            rec = dict(group="attn_6L_qkv", variant=name, M=m, ms=round(sec * 1e3, 4),
                       weight_MB=round(attn_bytes / 1e6, 2), TBps=round(tbps, 4),
                       pct_peak=round(tbps / 2.45 * 100, 2), rrmse=round(rrmse, 6))
            rows.append(rec)
            print(" ", rec, flush=True)

        x_lm = torch.randn(m, 4096, device="hpu", dtype=torch.bfloat16)
        sec = _bench(lambda: c_lm(x_lm), iters=15)
        tbps = lm_bytes / sec / 1e12
        rec = dict(group="lm_head", variant="fp8", M=m, ms=round(sec * 1e3, 4),
                   weight_MB=round(lm_bytes / 1e6, 2), TBps=round(tbps, 4),
                   pct_peak=round(tbps / 2.45 * 100, 2), rrmse=0.0)
        rows.append(rec)
        print(" ", rec, flush=True)

    (out_dir / "matrix-micro.json").write_text(json.dumps(rows, indent=2))
    del mlp_layers, gdn_layers, attn_layers, lm_w, lm_s
    gc.collect()
    torch.hpu.synchronize()
    return rows


def run_attention_micro(out_dir: Path):
    print("\n=== M0.B: MTP Tree Attention Microbenchmark (FusedSDPA vs kitchen tree_decode) ===", flush=True)
    torch.manual_seed(11)
    cap = 6899
    # 8 full attention layers of KV to avoid SRAM caching across calls
    ks = [torch.randn(1, 4, cap, 256, device="hpu", dtype=torch.bfloat16) * 0.2 for _ in range(8)]
    vs = [torch.randn(1, 4, cap, 256, device="hpu", dtype=torch.bfloat16) * 0.2 for _ in range(8)]

    _, tree_decode = hpu_kv_attention.compiled_ops()

    def fused_tree_sdpa(q, key, value, prefix_len, tree_mask_bits):
        # q: [1, 16, seq, 256], key/value: [1, 4, valid_cap, 256] (already contains written tree rows at prefix_len:prefix_len+seq)
        seq = q.shape[2]
        valid_cap = key.shape[2]
        rel = torch.arange(valid_cap, device=q.device).view(1, 1, 1, -1) - prefix_len.view(-1, 1, 1, 1)
        tree_vis = ((tree_mask_bits.view(1, 1, seq, 1) >> rel.clamp(0, seq - 1)) & 1).bool()
        visible = (rel < 0) | ((rel >= 0) & (rel < seq) & tree_vis)
        mask = torch.zeros(visible.shape, device=q.device, dtype=q.dtype).masked_fill(~visible, float("-inf"))
        out = FusedSDPA.apply(q, key, value, mask, 0.0, False, None, "fp32", False)
        return out.transpose(1, 2).reshape(q.shape[0], seq, -1)

    c_fused = torch.compile(fused_tree_sdpa, backend="hpu_backend", fullgraph=True, dynamic=None,
                            options={"force_static_compile": True})
    for k, v in zip(ks, vs):
        torch._dynamo.mark_dynamic(k, 2)
        torch._dynamo.mark_dynamic(v, 2)

    results = []
    for mode, spec, rows_n in [("chain4", [[[]]], 4), ("tree3s1", [[[[], []], []], []], 8)]:
        tree = qwen35.VerifyTree(spec, rows_n, torch.device("hpu"))
        qs = [torch.randn(1, 16, rows_n, 256, device="hpu", dtype=torch.bfloat16) * 0.2 for _ in range(8)]
        for prefix in (2803, 3800, 4700):
            seqlen = torch.tensor([prefix], device="hpu", dtype=torch.int32)
            xks = [k[:, :, prefix:prefix + rows_n].clone() for k in ks]
            xvs = [v[:, :, prefix:prefix + rows_n].clone() for v in vs]

            # Reference via kitchen tree_decode
            ref_outs = [tree_decode(q, k, v, seqlen, xk, xv, tree.mask).cpu().float()
                        for q, k, v, xk, xv in zip(qs, ks, vs, xks, xvs)]
            t_kitchen = _bench(lambda: [tree_decode(q, k, v, seqlen, xk, xv, tree.mask)
                                        for q, k, v, xk, xv in zip(qs, ks, vs, xks, xvs)], iters=20)
            results.append(dict(mode=mode, M=rows_n, prefix=prefix, variant="kitchen_tree_fullcap",
                                valid_cap=cap, total_8L_ms=round(t_kitchen * 1e3, 4),
                                per_layer_us=round(t_kitchen * 1e6 / 8, 2), rrmse_vs_kitchen=0.0))
            print(" ", results[-1], flush=True)

            # FusedSDPA full cap & bucketed dynamic slice (512-aligned)
            for vname, vcap in [("fused_sdpa_fullcap", cap),
                                ("fused_sdpa_dyn512", min(cap, ((prefix + rows_n + 511) // 512) * 512))]:
                k_slices = [k[:, :, :vcap] for k in ks]
                v_slices = [v[:, :, :vcap] for v in vs]
                for ks_i, vs_i in zip(k_slices, v_slices):
                    torch._dynamo.mark_dynamic(ks_i, 2)
                    torch._dynamo.mark_dynamic(vs_i, 2)
                outs = [c_fused(q, ks_i, vs_i, seqlen, tree.mask).cpu().float()
                        for q, ks_i, vs_i in zip(qs, k_slices, v_slices)]
                rrmse = max(((a - b).square().mean() / b.square().mean().clamp_min(1e-12)).sqrt().item()
                            for a, b in zip(outs, ref_outs))
                t_fused = _bench(lambda: [c_fused(q, ks_i, vs_i, seqlen, tree.mask)
                                          for q, ks_i, vs_i in zip(qs, k_slices, v_slices)], iters=20)
                results.append(dict(mode=mode, M=rows_n, prefix=prefix, variant=vname,
                                    valid_cap=vcap, total_8L_ms=round(t_fused * 1e3, 4),
                                    per_layer_us=round(t_fused * 1e6 / 8, 2), rrmse_vs_kitchen=round(rrmse, 6)))
                print(" ", results[-1], flush=True)

    (out_dir / "attention-micro.json").write_text(json.dumps(results, indent=2))
    del ks, vs
    gc.collect()
    torch.hpu.synchronize()
    return results


def run_mtp_screen(model_path: Path, label: str, out_dir: Path, modes, max_length=512, runs=2):
    print(f"\n=== M0.C: FP8 PE+MTP Baseline Screen ({label}, max_len={max_length}) ===", flush=True)
    wf_path = Path("/home/sonny/gaudi2/comfyui-hpu/workflows/qwen21-latest-2026-11-16-api.json")
    workflow = json.loads(wf_path.read_text())
    settings = workflow["459:471"]["inputs"]
    sys_prompt = workflow["459:475"]["inputs"]["value"]

    clip = comfy.sd.load_clip([str(model_path)], clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
    tokens = clip.tokenize(settings["prompt"], system_prompt=sys_prompt, min_length=1, thinking=settings["thinking"])
    clip.load_model(tokens)
    target = next(m for m in clip.cond_stage_model.modules() if isinstance(m, qwen35.Qwen35))
    assert target.mtp is not None, "Expected MTP head to be loaded automatically!"
    mtp_layouts = {name: getattr(m, "layout_type", None) for name, m in target.mtp.named_modules() if hasattr(m, "weight") and m.weight.ndim == 2}
    print("  MTP Linear layouts:", mtp_layouts, flush=True)

    counts = {}
    orig_fwd = target.model.forward
    orig_mtp = target.mtp.forward

    def wrap_fwd(*args, **kwargs):
        n = kwargs["embeds"].shape[1]
        k = "prefill" if n > 8 else f"verify_M{n}"
        counts[k] = counts.get(k, 0) + 1
        return orig_fwd(*args, **kwargs)

    def wrap_mtp(*args, **kwargs):
        counts["draft"] = counts.get("draft", 0) + 1
        return orig_mtp(*args, **kwargs)

    target.model.forward = wrap_fwd
    target.mtp.forward = wrap_mtp

    records = []
    for mode in modes:
        for turn in range(runs):
            counts.clear()
            gc.collect()
            torch.hpu.synchronize()
            t0 = time.perf_counter()
            ids = clip.generate(
                tokens, do_sample=True, max_length=max_length,
                temperature=1.0, top_k=20, top_p=0.95, min_p=0.0,
                repetition_penalty=1.0, presence_penalty=1.5,
                seed=settings["sampling_mode.seed"],
                mtp=False if mode == "off" else (True if mode == "auto" else mode),
            )
            torch.hpu.synchronize()
            dt = time.perf_counter() - t0
            vcalls = sum(v for k, v in counts.items() if k.startswith("verify_M"))
            rec = dict(
                label=label, mode=mode, turn=turn, cold=(turn == 0),
                seconds=round(dt, 4), tokens=len(ids), tok_s=round(len(ids) / dt, 2),
                counts=dict(counts),
                accepted_per_verify=round((len(ids) - 1) / vcalls, 3) if vcalls else 0.0,
                drafts_per_verify=round(counts.get("draft", 0) / vcalls, 3) if vcalls else 0.0,
                mtp_layouts=mtp_layouts,
            )
            records.append(rec)
            print(" ", rec, flush=True)
            (out_dir / f"{label}-{mode}-t{turn}.txt").write_text(clip.decode(ids))

    (out_dir / f"screen-{label}.json").write_text(json.dumps(records, indent=2))
    del clip, target
    gc.collect()
    torch.hpu.synchronize()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--part", choices=["all", "micro", "screen-fp8", "screen-bf16"], default="all")
    parser.add_argument("--fp8-mtp", type=Path, required=True)
    parser.add_argument("--bf16-mtp", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--modes", nargs="+", default=["2", "3", "tree3s1", "auto"])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(32)
    torch._dynamo.config.recompile_limit = 64

    if args.part in ("all", "micro"):
        run_matrix_micro(args.fp8_mtp, args.out_dir)
        run_attention_micro(args.out_dir)
    if args.part in ("all", "screen-fp8"):
        run_mtp_screen(args.fp8_mtp, "mtp_fp8", args.out_dir, args.modes, max_length=args.max_length, runs=2)
    if args.part in ("all", "screen-bf16") and args.bf16_mtp is not None:
        run_mtp_screen(args.bf16_mtp, "mtp_bf16", args.out_dir, args.modes, max_length=args.max_length, runs=2)


if __name__ == "__main__":
    main()
