"""Screen segmented Qwen blocks on captured positive/negative inputs, layers 0, 15 and 31."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time
import torch
import torch.nn.functional as F
from comfy.cli_args import args as comfy_args
comfy_args.disable_dynamic_vram = True
comfy_args.use_pytorch_cross_attention = True
import comfy.sd
import comfy.model_management
import comfy.ops
import comfy.quant_ops
from comfy.ldm.qwen_image21 import model as dit
from comfy.ldm.modules import hpu_attention


def get_qweight(linear):
    w = getattr(linear, "_v_weight", None)
    if w is None:
        w = linear.weight
    return w._qdata, w._params.scale


def get_norm_weight(norm):
    w = getattr(norm, "_v_weight", None)
    if w is None:
        w = norm.weight
    return w.to(torch.bfloat16)


def diff_stat(a, b):
    ac, bc = a.cpu(), b.cpu()
    return {
        "bitwise": bool(torch.equal(ac, bc)),
        "rrmse": float((bc.float() - ac.float()).norm() / ac.float().norm()),
        "max_abs": float((bc.float() - ac.float()).abs().max()),
    }


def f32_barrier(t: torch.Tensor) -> torch.Tensor:
    return t.float().view(torch.int32).view(torch.float32)


def bf16_barrier(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).view(torch.int16).view(torch.bfloat16)


def bf16_round_via_f32(t: torch.Tensor) -> torch.Tensor:
    return f32_barrier(t).to(torch.bfloat16).view(torch.int16).view(torch.bfloat16)


def exact_addcmul(x: torch.Tensor, y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    prod = bf16_round_via_f32(y * g)
    return bf16_round_via_f32(x + prod)


divisor = 240.0


def exact_fp8_quant(flat):
    flat = bf16_round_via_f32(flat)
    amax = bf16_round_via_f32(flat.abs().amax(-1, keepdim=True))
    scale = f32_barrier(amax.float().clamp_min(1e-12) / divisor)
    inv_bf16 = bf16_barrier(f32_barrier(scale.reciprocal()))
    scaled = bf16_round_via_f32(flat * inv_bf16)
    qx = torch.ops.hpu.cast_to_fp8_v2(scaled, None, False, False, torch.float8_e4m3fn)[0]
    return qx, scale


def exact_fp8_gemm(qx, scale, out_shape, weight, weight_scale, out_dtype):
    unscaled = torch.ops.hpu.fp8_gemm_v2(qx, False, weight, True, None, out_dtype, None, None, None, False)
    comb_bf16 = bf16_barrier(f32_barrier(scale * weight_scale))
    return bf16_round_via_f32(unscaled * comb_bf16).view(*out_shape)


def exact_fp8_linear(x, weight, weight_scale):
    flat = x.reshape(-1, x.shape[-1])
    qx, scale = exact_fp8_quant(flat)
    return exact_fp8_gemm(qx, scale, (*x.shape[:-1], weight.shape[0]), weight, weight_scale, x.dtype)


def _compiled_fp8_linear(input, weight, weight_scale):
    flat = bf16_round_via_f32(input).reshape(-1, input.shape[-1])
    input_scale = flat.abs().amax(-1, keepdim=True).float().clamp_min(1e-12) / 240.0
    quantized = torch.ops.hpu.cast_to_fp8_v2(flat, input_scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
    output = torch.ops.hpu.fp8_gemm_v2(quantized, False, weight, True, None, input.dtype,
                                       input_scale, weight_scale, None, False)
    return bf16_round_via_f32(output).reshape(*input.shape[:-1], weight.shape[0])


def step_qkv_exact(normed, wq, sq, wk, sk, wv, sv):
    B, N, _ = normed.shape
    flat = normed.reshape(-1, normed.shape[-1])
    qx, scale = exact_fp8_quant(flat)
    q = exact_fp8_gemm(qx, scale, (B, N, 32, 128), wq, sq, normed.dtype)
    k = exact_fp8_gemm(qx, scale, (B, N, 32, 128), wk, sk, normed.dtype)
    v = exact_fp8_gemm(qx, scale, (B, N, 32, 128), wv, sv, normed.dtype)
    return q, k, v


def _pre_attn_stage1(x, s1, zero, eps1, wq, sq, wk, sk, wv, sv, nq_w, nk_w, eps_q):
    normed = comfy.quant_ops.ck.adaln(x, s1, zero, eps1)
    q, k, v = step_qkv_exact(normed, wq, sq, wk, sk, wv, sv)
    qn = F.rms_norm(q, (q.shape[-1],), nq_w, eps_q)
    kn = F.rms_norm(k, (k.shape[-1],), nk_w, eps_q)
    return qn, kn, v


def _pre_attn_stage2_split(q, k, v, cos, sin, prefix_k, prefix_v):
    q = torch.ops.hpu.rotary_pos_embedding(q.to(cos.dtype), sin, cos, None, 0, 1).to(q.dtype)
    k = torch.ops.hpu.rotary_pos_embedding(k.to(cos.dtype), sin, cos, None, 0, 1).to(k.dtype)
    k_full = torch.cat([prefix_k, k], dim=1).transpose(1, 2)
    v_full = torch.cat([prefix_v, v], dim=1).transpose(1, 2)
    q_t = q.transpose(1, 2)
    split = (k_full.shape[-2] // 128) * 128
    scale = torch.full((), q_t.shape[-1] ** -0.5, dtype=q_t.dtype, device=q_t.device)
    qh = (q_t * scale).contiguous()
    kh = (k_full.float() - k_full.float().mean(dim=-2, keepdim=True)).to(k_full.dtype)
    k1 = kh[..., :split, :].contiguous()
    v1 = v_full[..., :split, :].contiguous()
    k2 = kh[..., split:, :].contiguous()
    v2 = v_full[..., split:, :].contiguous()
    last, max2, inv2 = torch.ops.hpu.sdpa_recomp_fwd(
        qh, k2, v2, None, 0.0, 1.0, False, True, "fast", None, "right"
    )[:3]
    return qh, k1, v1, last, max2, inv2


def _pre_attn_stage2_aligned(q, k, v, cos, sin, prefix_k, prefix_v):
    q = torch.ops.hpu.rotary_pos_embedding(q.to(cos.dtype), sin, cos, None, 0, 1).to(q.dtype)
    k = torch.ops.hpu.rotary_pos_embedding(k.to(cos.dtype), sin, cos, None, 0, 1).to(k.dtype)
    k_full = torch.cat([prefix_k, k], dim=1).transpose(1, 2)
    v_full = torch.cat([prefix_v, v], dim=1).transpose(1, 2)
    q_t = q.transpose(1, 2)
    scale = torch.full((), q_t.shape[-1] ** -0.5, dtype=q_t.dtype, device=q_t.device)
    qh = (q_t * scale).contiguous()
    kh = (k_full.float() - k_full.float().mean(dim=-2, keepdim=True)).to(k_full.dtype).contiguous()
    vh = v_full.contiguous()
    return qh, kh, vh


def _post_attn_tail_and_mlp(attn_out, x, gate1, scale2, gate2, zero, eps2, w_out, s_out, w_gu, s_gu, w_down, s_down):
    proj_out = exact_fp8_linear(attn_out, w_out, s_out)
    x_mid = exact_addcmul(x, proj_out, gate1)
    normed2 = comfy.quant_ops.ck.adaln(x_mid, scale2, zero, eps2)
    gu = _compiled_fp8_linear(normed2, w_gu, s_gu)
    gate, up = gu.chunk(2, dim=-1)
    act = bf16_round_via_f32(F.silu(gate) * up)
    mlp_out = _compiled_fp8_linear(act, w_down, s_down)
    return exact_addcmul(x_mid, mlp_out, gate2)


def _post_attn_split_and_mlp(first, max1, inv1, last, max2, inv2, x, gate1, scale2, gate2, zero, eps2, w_out, s_out, w_gu, s_gu, w_down, s_down):
    m1 = max1.float()
    m2 = max2.float()
    maximum = torch.maximum(m1, m2)
    w1 = (m1 - maximum).exp() / inv1.float()
    w2 = (m2 - maximum).exp() / inv2.float()
    share = (w2 / (w1 + w2)).to(first.dtype)
    merged = torch.addcmul(first, last - first, share)
    attn_out = merged.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
    return _post_attn_tail_and_mlp(attn_out, x, gate1, scale2, gate2, zero, eps2, w_out, s_out, w_gu, s_gu, w_down, s_down)


def _post_attn_aligned_and_mlp(out, x, gate1, scale2, gate2, zero, eps2, w_out, s_out, w_gu, s_gu, w_down, s_down):
    attn_out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
    return _post_attn_tail_and_mlp(attn_out, x, gate1, scale2, gate2, zero, eps2, w_out, s_out, w_gu, s_gu, w_down, s_down)


def baseline_forward(block, x, mod, pe, prefix_k, prefix_v, layer_idx):
    settings = dict(hpu_fast_attention=True, block_index=layer_idx)
    attn_fn = dit.prefix_cached_attention(prefix_k, prefix_v, settings)
    return block(x, mod, pe, attn_fn, 0, settings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tensor-divisor", action="store_true")
    options = parser.parse_args()
    if options.tensor_divisor:
        global divisor
        divisor = torch.tensor(240.0, device="hpu")
    options.output.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(32)

    patcher = comfy.sd.load_diffusion_model(
        "/home/sonny/ComfyUI-Models/models/diffusion_models/qwen_image_2.1_turbo_fp8.safetensors"
    )
    comfy.model_management.load_models_gpu([patcher])
    torch.hpu.synchronize()

    opts = {"force_static_compile": True}
    c_pre1 = torch.compile(_pre_attn_stage1, backend="hpu_backend", dynamic=None, options=opts)
    c_pre2_split = torch.compile(_pre_attn_stage2_split, backend="hpu_backend", dynamic=None, options=opts)
    c_pre2_aligned = torch.compile(_pre_attn_stage2_aligned, backend="hpu_backend", dynamic=None, options=opts)
    c_post_split = torch.compile(_post_attn_split_and_mlp, backend="hpu_backend", dynamic=None, options=opts)
    c_post_aligned = torch.compile(_post_attn_aligned_and_mlp, backend="hpu_backend", dynamic=None, options=opts)
    c_post_layer0 = torch.compile(_post_attn_tail_and_mlp, backend="hpu_backend", dynamic=None, options=opts)

    rt = hpu_attention._runtime()

    def run_plan_b_block(block, x, mod, cos, sin, prefix_k, prefix_v, layer_idx):
        scale1, gate1, scale2, gate2, zero = mod
        attn = block.attn
        wq, sq = get_qweight(attn.to_q)
        wk, sk = get_qweight(attn.to_k)
        wv, sv = get_qweight(attn.to_v)
        w_out, s_out = get_qweight(attn.to_out[0])
        w_gu, s_gu = get_qweight(block.img_mlp.gate_up)
        w_down, s_down = get_qweight(block.img_mlp.out)
        nq_w = get_norm_weight(attn.norm_q)
        nk_w = get_norm_weight(attn.norm_k)

        qn, kn, v = c_pre1(
            x, scale1[1], zero, block.img_norm1.eps,
            wq, sq, wk, sk, wv, sv, nq_w, nk_w, attn.norm_q.eps
        )
        if layer_idx == 0:
            qr = torch.ops.hpu.rotary_pos_embedding(qn.to(cos.dtype), sin, cos, None, 0, 1).to(qn.dtype)
            kr = torch.ops.hpu.rotary_pos_embedding(kn.to(cos.dtype), sin, cos, None, 0, 1).to(kn.dtype)
            settings = dict(hpu_fast_attention=False, block_index=0)
            attn_out = dit._compiled_attention()(
                qr.flatten(2),
                torch.cat([prefix_k, kr], dim=1).flatten(2),
                torch.cat([prefix_v, v], dim=1).flatten(2),
                32, None, settings, attn.comfy_attention
            )
            return c_post_layer0(
                attn_out, x, gate1[1], scale2[1], gate2[1], zero, block.img_norm2.eps,
                w_out, s_out, w_gu, s_gu, w_down, s_down
            )

        torch._dynamo.mark_dynamic(prefix_k, 1)
        torch._dynamo.mark_dynamic(prefix_v, 1)
        queries = x.shape[1]
        full_keys = prefix_k.shape[1] + queries
        aligned_keys = (full_keys // 128) * 128
        retained = 0 if aligned_keys == full_keys else 1
        plan = rt.get_plan(queries, aligned_keys, retained)
        ws = rt.get_workspace(int(rt.lib.rmw_plan_workspace_bytes(plan)), x.device)

        if not retained:
            qh, kh, vh = c_pre2_aligned(qn, kn, v, cos, sin, prefix_k, prefix_v)
            out = torch.empty_like(qh)
            rt._debug_eager_C.join_pending_pipeline_threads()
            ptrs = (hpu_attention.C.c_uint64 * 6)(
                rt._address(qh), rt._address(kh), rt._address(vh), rt._address(out), 0, 0
            )
            st = rt.lib.launch_rmw_plan(plan, rt._compute_stream(), ptrs, rt._address(ws), 32)
            assert st == 0, st
            return c_post_aligned(
                out, x, gate1[1], scale2[1], gate2[1], zero, block.img_norm2.eps,
                w_out, s_out, w_gu, s_gu, w_down, s_down
            )

        qh, k1, v1, last, max2, inv2 = c_pre2_split(qn, kn, v, cos, sin, prefix_k, prefix_v)
        first = torch.empty_like(qh)
        stat_queries = ((queries + 63) // 64) * 64
        max1 = torch.empty(1, 32, stat_queries, 1, device=qh.device, dtype=qh.dtype)
        inv1 = torch.empty_like(max1)
        rt._debug_eager_C.join_pending_pipeline_threads()
        ptrs = (hpu_attention.C.c_uint64 * 6)(
            rt._address(qh), rt._address(k1), rt._address(v1),
            rt._address(first), rt._address(max1), rt._address(inv1)
        )
        st = rt.lib.launch_rmw_plan(plan, rt._compute_stream(), ptrs, rt._address(ws), 32)
        assert st == 0, st
        return c_post_split(
            first, max1[:, :, :queries], inv1[:, :, :queries], last, max2, inv2,
            x, gate1[1], scale2[1], gate2[1], zero, block.img_norm2.eps,
            w_out, s_out, w_gu, s_gu, w_down, s_down
        )

    files = sorted(options.inputs.glob("forward-*.pt"))
    results = {}
    for f in files:
        saved = torch.load(f, weights_only=True)
        layer = saved["layer"]
        block = patcher.model.diffusion_model.transformer_blocks[layer]
        x, pe, prefix_k, prefix_v = [saved[name].to("hpu") for name in ("x", "pe", "prefix_k", "prefix_v")]
        mod = tuple(tuple(t.to("hpu") for t in item) if isinstance(item, tuple) else item.to("hpu") for item in saved["mod"])
        cos = torch.stack((pe[..., 0, 0], pe[..., 1, 1]), dim=-1).flatten(-2)
        sin = torch.stack((-pe[..., 0, 1], pe[..., 1, 0]), dim=-1).flatten(-2)

        ref_out = baseline_forward(block, x.clone(), mod, pe, prefix_k, prefix_v, layer)
        b_out = run_plan_b_block(block, x.clone(), mod, cos, sin, prefix_k, prefix_v, layer)
        torch.hpu.synchronize()

        for _ in range(2):
            baseline_forward(block, x.clone(), mod, pe, prefix_k, prefix_v, layer)
            run_plan_b_block(block, x.clone(), mod, cos, sin, prefix_k, prefix_v, layer)
        torch.hpu.synchronize()

        times = {"plan_a": [], "plan_b": []}
        for r in range(7):
            order = [
                ("plan_a", lambda: baseline_forward(block, x.clone(), mod, pe, prefix_k, prefix_v, layer)),
                ("plan_b", lambda: run_plan_b_block(block, x.clone(), mod, cos, sin, prefix_k, prefix_v, layer)),
            ]
            if r % 2 == 1:
                order.reverse()
            for m, fn in order:
                gc.collect()
                torch.hpu.synchronize()
                t0 = time.perf_counter()
                for _ in range(5):
                    fn()
                torch.hpu.synchronize()
                times[m].append((time.perf_counter() - t0) / 5.0 * 1000.0)

        results[f.name] = {
            "layer": layer,
            "diff_vs_plan_a": diff_stat(ref_out, b_out),
            "median_ms": {k: statistics.median(v) for k, v in times.items()},
            "speedup_ms": statistics.median(times["plan_a"]) - statistics.median(times["plan_b"]),
        }

    print(json.dumps(results, indent=2), flush=True)
    (options.output / ("barrier-audit8.json" if options.tensor_divisor else "barrier-audit7.json")).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
