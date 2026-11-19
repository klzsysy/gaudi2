"""M1+M2 correctness check, ablation benchmark, and full-length natural PE+MTP timing on HPU."""
import argparse
import gc
import json
import statistics
import time
import warnings
from pathlib import Path

import torch
import habana_frameworks.torch

from comfy.cli_args import args as comfy_args
comfy_args.disable_dynamic_vram = True
comfy_args.disable_pinned_memory = True
comfy_args.disable_async_offload = True
comfy_args.use_pytorch_cross_attention = True

import comfy.ops
import comfy.sd
from comfy.text_encoders import hpu_qwen35, qwen35


def run_synthetic_correctness():
    print("=== 1. Synthetic 4L FP8+MTP Correctness & Zero-Recompile Check ===", flush=True)
    torch.manual_seed(123)
    cfg = qwen35.Qwen35Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=256,
        linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=qwen35._qwen35_layer_types(4), lm_head=True, mtp=True,
    )
    # Build BF16 reference then quantize to FP8
    base = qwen35.Qwen35.__new__(qwen35.Qwen35)
    torch.nn.Module.__init__(base)
    base.model = qwen35.Qwen35Transformer(cfg, device="cpu", dtype=torch.bfloat16, ops=comfy.ops.manual_cast)
    base.mtp = qwen35.MTPHead(cfg, device="cpu", dtype=torch.bfloat16, ops=comfy.ops.manual_cast)
    rng = torch.Generator().manual_seed(99)
    for p in list(base.model.parameters()) + list(base.mtp.parameters()):
        p.data.copy_(torch.randn(p.shape, generator=rng) * 0.04)

    state = {}
    for prefix, mod in [("model.", base.model), ("mtp.", base.mtp)]:
        sd = mod.state_dict()
        for name, sub in mod.named_modules():
            if isinstance(sub, torch.nn.Linear) and not name.endswith(("in_proj_a", "in_proj_b")):
                w = sd[name + ".weight"].float()
                scale = w.abs().amax().clamp_min(1e-12) / 224.0
                sd[name + ".weight"] = (w / scale).to(torch.float8_e4m3fn)
                sd[name + ".weight_scale"] = scale
                sd[name + ".comfy_quant"] = torch.tensor(list(json.dumps({"format": "float8_e4m3fn"}).encode()), dtype=torch.uint8)
        for k, v in sd.items():
            state[prefix + k] = v

    ops = comfy.ops.mixed_precision_ops({"formats": ["float8_e4m3fn"]}, torch.bfloat16, load_device=torch.device("hpu"))
    q = qwen35.Qwen35.__new__(qwen35.Qwen35)
    torch.nn.Module.__init__(q)
    q.model = qwen35.Qwen35Transformer(cfg, device="cpu", dtype=torch.bfloat16, ops=ops)
    q.mtp = qwen35.MTPHead(cfg, device="cpu", dtype=torch.bfloat16, ops=ops)
    q.model.load_state_dict({k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")})
    q.mtp.load_state_dict({k.removeprefix("mtp."): v for k, v in state.items() if k.startswith("mtp.")})
    q.to("hpu")

    prompt = torch.arange(1, 21, device="hpu").unsqueeze(0)
    prompt_b = prompt.clone()
    prompt_b[:, -1] = 77
    embeds_a = q.model.embed_tokens(prompt)
    embeds_b = q.model.embed_tokens(prompt_b)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # 1a. Compare compiled MTP vs disabled-compiler eager MTP (greedy) across modes
        for mode in ("2", "tree3s1", True):
            comfy_args.disable_comfy_compiler = True
            ids_eager = q.generate(embeds=embeds_a.clone(), do_sample=False, max_length=16, stop_tokens=[], seed=7, mtp=mode)
            comfy_args.disable_comfy_compiler = False
            ids_comp = q.generate(embeds=embeds_a.clone(), do_sample=False, max_length=16, stop_tokens=[], seed=7, mtp=mode)
            match = (ids_eager == ids_comp)
            print(f"  Greedy mode={mode!r}: eager==compiled? {match} eager={ids_eager[:10]} comp={ids_comp[:10]}", flush=True)
            assert ids_eager[0] == ids_comp[0], f"Prefill token mismatch: {ids_eager[0]} vs {ids_comp[0]}"
            # Sampled mode with presence_penalty + A->B->A isolation
            s_a1 = q.generate(embeds=embeds_a.clone(), do_sample=True, max_length=16, temperature=1.0, top_k=10, top_p=0.95, presence_penalty=1.5, stop_tokens=[], seed=42, mtp=mode)
            s_b1 = q.generate(embeds=embeds_b.clone(), do_sample=True, max_length=16, temperature=1.0, top_k=10, top_p=0.95, presence_penalty=1.5, stop_tokens=[], seed=42, mtp=mode)
            s_a2 = q.generate(embeds=embeds_a.clone(), do_sample=True, max_length=16, temperature=1.0, top_k=10, top_p=0.95, presence_penalty=1.5, stop_tokens=[], seed=42, mtp=mode)
            assert s_a1 == s_a2, f"A->B->A mismatch for mode={mode}: {s_a1} vs {s_a2}"
            assert s_a1 != s_b1, f"Prompt B unexpectedly matched Prompt A for mode={mode}"
            print(f"  Sampled mode={mode!r}: A->B->A exact match OK (len={len(s_a1)})", flush=True)

        # 1b. Test dynamic capacity reuse across varying prompt lengths
        for plen in (12, 17, 25, 33, 64, 12):
            p_var = torch.arange(1, plen + 1, device="hpu").unsqueeze(0)
            out_v = q.generate(embeds=q.model.embed_tokens(p_var), do_sample=True, max_length=12,
                               temperature=1.0, top_k=10, top_p=0.95, presence_penalty=1.5, stop_tokens=[], seed=42, mtp="tree3s1")
            assert len(out_v) == 12
        # 1c. Test mtp=False on MTP model uses fast single-token path cleanly
        off_a1 = q.generate(embeds=embeds_a.clone(), do_sample=True, max_length=12, temperature=1.0, top_k=10, top_p=0.95, presence_penalty=1.5, stop_tokens=[], seed=42, mtp=False)
        off_a2 = q.generate(embeds=embeds_a.clone(), do_sample=True, max_length=12, temperature=1.0, top_k=10, top_p=0.95, presence_penalty=1.5, stop_tokens=[], seed=42, mtp=False)
        assert off_a1 == off_a2
        print("  Dynamic capacity & mtp=False on MTP model OK!", flush=True)

    alias_warns = [str(w.message) for w in caught if "batch_as_strided" in str(w.message)]
    recomp_warns = [str(w.message) for w in caught if "recompile_limit" in str(w.message)]
    print(f"  Warnings: alias={len(alias_warns)}, recompile_limit={len(recomp_warns)}", flush=True)
    del q, base
    torch._dynamo.reset()
    hpu_qwen35.reset_compile_caches()
    qwen35._compiled_hpu_decode.cache_clear()
    gc.collect()
    torch.hpu.synchronize()


def run_real_screen(model_path: Path, label: str, out_dir: Path, modes, max_length: int, runs: int,
                    shared_quant: bool, use_fused_sdpa: bool, kv_bucket: int, fuse_stage4: bool = False):
    print(f"\n=== Real 9B Screen: {label} (shared={shared_quant}, sdpa={use_fused_sdpa}, bucket={kv_bucket}, stage4={fuse_stage4}, max_len={max_length}) ===", flush=True)
    wf_path = Path("/home/sonny/gaudi2/comfyui-hpu/workflows/qwen21-latest-2026-11-16-api.json")
    workflow = json.loads(wf_path.read_text())
    settings = workflow["459:471"]["inputs"]
    sys_prompt = workflow["459:475"]["inputs"]["value"]

    orig_builder = hpu_qwen35.build_mtp_runner

    def patched_builder(*args, **kwargs):
        kwargs["shared_quant"] = shared_quant
        kwargs["use_fused_sdpa"] = use_fused_sdpa
        kwargs["kv_bucket"] = kv_bucket
        kwargs["fuse_stage4"] = fuse_stage4
        return orig_builder(*args, **kwargs)

    hpu_qwen35.build_mtp_runner = patched_builder
    torch._dynamo.reset()
    hpu_qwen35.reset_compile_caches()
    qwen35._compiled_hpu_decode.cache_clear()

    clip = comfy.sd.load_clip([str(model_path)], clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
    tokens = clip.tokenize(settings["prompt"], system_prompt=sys_prompt, min_length=1, thinking=settings["thinking"])
    clip.load_model(tokens)

    # Also test prompt-edit dynamic startup latency (adding 1 word changes capacity!)
    tokens_edit = clip.tokenize(settings["prompt"] + " 柔和光线。", system_prompt=sys_prompt, min_length=1, thinking=settings["thinking"])

    records = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for mode in modes:
            for turn in range(runs):
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
                rec = dict(label=label, mode=mode, turn=turn, cold=(turn == 0),
                           seconds=round(dt, 4), tokens=len(ids), tok_s=round(len(ids) / dt, 2),
                           shared_quant=shared_quant, use_fused_sdpa=use_fused_sdpa, kv_bucket=kv_bucket)
                records.append(rec)
                print(" ", rec, flush=True)
                (out_dir / f"{label}-{mode}-t{turn}.txt").write_text(clip.decode(ids))

        # Quick dynamic capacity check on edited prompt (short 64 tokens to check startup speed)
        if "auto" in modes or "tree3s1" in modes:
            m_chk = "auto" if "auto" in modes else "tree3s1"
            torch.hpu.synchronize()
            t0 = time.perf_counter()
            ids_ed = clip.generate(
                tokens_edit, do_sample=True, max_length=64,
                temperature=1.0, top_k=20, top_p=0.95, min_p=0.0,
                repetition_penalty=1.0, presence_penalty=1.5,
                seed=settings["sampling_mode.seed"],
                mtp=True if m_chk == "auto" else m_chk,
            )
            torch.hpu.synchronize()
            dt_ed = time.perf_counter() - t0
            rec_ed = dict(label=label, mode=f"{m_chk}_edit_dyn", turn=0, seconds=round(dt_ed, 4),
                          tokens=len(ids_ed), tok_s=round(len(ids_ed) / dt_ed, 2))
            records.append(rec_ed)
            print("  DYNAMIC PROMPT EDIT CHECK:", rec_ed, flush=True)

    alias_warns = [str(w.message) for w in caught if "batch_as_strided" in str(w.message)]
    recomp_warns = [str(w.message) for w in caught if "recompile_limit" in str(w.message)]
    mem = torch.hpu.memory_stats()
    summary = dict(records=records, alias_warnings=len(alias_warns), recompile_warnings=len(recomp_warns),
                   in_use_gib=round(mem["InUse"] / 1024**3, 3), max_in_use_gib=round(mem["MaxInUse"] / 1024**3, 3))
    print(f"  Summary: alias={len(alias_warns)}, recompile={len(recomp_warns)}, InUse={summary['in_use_gib']}GiB, MaxInUse={summary['max_in_use_gib']}GiB", flush=True)
    (out_dir / f"{label}.json").write_text(json.dumps(summary, indent=2))

    hpu_qwen35.build_mtp_runner = orig_builder
    del clip
    gc.collect()
    torch.hpu.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--part", choices=["synthetic", "screen"], required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--label", default="m2")
    parser.add_argument("--modes", nargs="+", default=["auto", "tree3s1", "2", "off"])
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--no-shared-quant", action="store_true")
    parser.add_argument("--no-fused-sdpa", action="store_true")
    parser.add_argument("--kv-bucket", type=int, default=0)
    parser.add_argument("--fuse-stage4", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(32)

    if args.part == "synthetic":
        run_synthetic_correctness()
    else:
        run_real_screen(
            args.model, args.label, args.out_dir, args.modes, args.max_length, args.runs,
            shared_quant=not args.no_shared_quant,
            use_fused_sdpa=not args.no_fused_sdpa,
            kv_bucket=args.kv_bucket,
            fuse_stage4=args.fuse_stage4,
        )


if __name__ == "__main__":
    main()
