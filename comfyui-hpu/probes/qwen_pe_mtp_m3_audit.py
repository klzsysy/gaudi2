"""Low-perturbation BMON HBM, native timeline, telemetry, and acceptance audit for Qwen3.5 FP8 PE+MTP M3."""
import argparse
import collections
import gc
import json
import os
import statistics
import subprocess
import sys
import threading
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

import comfy.sd
from comfy.text_encoders import hpu_qwen35, hpu_qwen35_tpc, qwen35

sys.path.insert(0, str(Path(__file__).parent))


def sample_hl_smi(stop_event: threading.Event, out_list: list, interval: float = 0.25):
    query = "power.draw,temperature.aip,utilization.aip,clocks.current.soc"
    while not stop_event.is_set():
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                ["hl-smi", f"--query-aip={query}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2.0,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
                out_list.append({
                    "t": round(time.perf_counter(), 3),
                    "power_w": float(parts[0]) if len(parts) > 0 else None,
                    "temp_c": float(parts[1]) if len(parts) > 1 else None,
                    "util_pct": float(parts[2]) if len(parts) > 2 else None,
                    "clock_mhz": float(parts[3]) if len(parts) > 3 else None,
                })
        except Exception:
            pass
        rem = interval - (time.perf_counter() - t0)
        if rem > 0:
            stop_event.wait(rem)


def summarize_bmon(profiler_json: Path, events_jsonl: Path, profiled_s: float, plain_s: float,
                   verify_steps: int, accepted_tokens: int):
    config = json.loads(profiler_json.read_text())
    hw = next(p for p in config["Plugins"] if p["name"] == "HwTrace")
    bmon = hw["values"]["archProfileUnits"]["gaudi2"]["HMMU"]["DCORE0_HMMU0_CS"]["bmonParams"]["b10"]
    factor = (bmon["capture_window"]["value"] + 1) / (bmon["bandwidth_window"]["value"] + 1) * 128
    ports = collections.defaultdict(lambda: dict(samples=0, bytes=0, rounding_envelope=0))
    for line in events_jsonl.open():
        event = json.loads(line)
        if event.get("category") != "BMON" or "counter_values" not in event:
            continue
        port = ports[event["name"]]
        port["samples"] += 1
        port["bytes"] += event["counter_values"][2] * factor
        port["rounding_envelope"] += (event["counter_values"][1] > 0) * factor
    read_b = sum(v["bytes"] for k, v in ports.items() if "Read" in k)
    write_b = sum(v["bytes"] for k, v in ports.items() if "Write" in k)
    total_b = read_b + write_b
    return {
        "read_bytes": read_b,
        "write_bytes": write_b,
        "total_bytes": total_b,
        "verify_steps": verify_steps,
        "accepted_tokens": accepted_tokens,
        "acc_per_verify": round(accepted_tokens / max(1, verify_steps), 4),
        "gb_per_verify": round(total_b / max(1, verify_steps) / 1e9, 4),
        "gb_per_token": round(total_b / max(1, accepted_tokens) / 1e9, 4),
        "profiled_seconds": round(profiled_s, 6),
        "plain_seconds": round(plain_s, 6),
        "profiler_slowdown_pct": round((profiled_s / plain_s - 1.0) * 100.0, 3),
        "profiled_TBps": round(total_b / profiled_s / 1e12, 4),
        "plain_estimated_TBps": round(total_b / plain_s / 1e12, 4),
        "bandwidth_pct_245": round((total_b / plain_s / 2.45e12) * 100.0, 2),
        "ports_seen": len(ports),
    }


def summarize_native_timeline(events_jsonl: Path, verify_steps: int, accepted_tokens: int):
    def union(intervals):
        res = []
        for s, e in sorted(intervals):
            if res and s <= res[-1][1]:
                res[-1][1] = max(e, res[-1][1])
            else:
                res.append([s, e])
        return res

    def dur(intervals):
        return sum(e - s for s, e in intervals)

    stacks = collections.defaultdict(list)
    engines = {}
    events = collections.defaultdict(list)
    recipes = []
    for line in events_jsonl.open():
        ev = json.loads(line)
        pid, ph, ts = ev["pid"], ev["ph"], ev["ts"]
        if ph == "M" and ev["name"] == "process_name":
            name = ev.get("metadata_name", "").split(" ")[0].lstrip("*")
            if name in ("MME", "TPC", "EDMA", "PDMA"):
                engines[pid] = name
        if ev.get("category") == "Recipe Start":
            recipes.append((ts, ts + ev["dur"]))
        if pid not in engines or ph not in ("B", "E"):
            continue
        key = (pid, ev["tid"], ev["id"])
        if ph == "B":
            stacks[key].append(ts)
        elif stacks[key]:
            start = stacks[key].pop()
            if ts >= start:
                events[engines[pid]].append((start, ts))
    compute = union(events["MME"] + events["TPC"] + events["EDMA"])
    span = (compute[-1][1] - compute[0][0]) if compute else 0
    covered = dur(compute)
    return {
        "engines": {k: {"events": len(v), "covered_ms": round(dur(union(v)) / 1000.0, 3)} for k, v in events.items()},
        "compute_span_ms": round(span / 1000.0, 3),
        "compute_covered_ms": round(covered / 1000.0, 3),
        "compute_uncovered_ms": round((span - covered) / 1000.0, 3),
        "compute_covered_pct": round(100.0 * covered / max(1.0, span), 2),
        "compute_gap_pct": round(100.0 * (span - covered) / max(1.0, span), 2),
        "recipe_count": len(recipes),
        "recipes_per_verify": round(len(recipes) / max(1, verify_steps), 3),
        "recipes_per_token": round(len(recipes) / max(1, accepted_tokens), 3),
    }


def run_full_bench(args):
    print(f"hpu_qwen35_tpc.available={hpu_qwen35_tpc.available}", flush=True)
    wf = json.loads(Path(args.workflow).read_text())
    settings = wf["459:471"]["inputs"]
    sys_prompt = wf["459:475"]["inputs"]["value"]

    orig_builder = hpu_qwen35.build_mtp_runner
    stats = {"verify_steps": 0, "draft_steps": 0, "runner_info": {}}

    def wrapped_builder(*bargs, **bkwargs):
        bkwargs["shared_quant"] = not args.no_shared_quant
        bkwargs["use_fused_sdpa"] = not args.no_fused_sdpa
        bkwargs["kv_bucket"] = args.kv_bucket
        bkwargs["fuse_stage4"] = args.stage4
        if args.no_tpc:
            bkwargs["use_tpc"] = False
        if args.batch_steps is not None:
            bkwargs["batch_steps"] = args.batch_steps
        res = orig_builder(*bargs, **bkwargs)
        if res is None:
            return None
        stats["runner_info"] = {
            k: res[k] for k in (
                "use_tpc_gdn", "use_tpc_rms", "use_tpc_topk", "use_tpc_gated",
                "use_tpc_silu", "use_tpc_sig", "use_tpc_qk", "use_tpc_accept",
                "batch_steps",
            ) if k in res
        }
        stats["runner_info"]["can_batch"] = res["run_batch"] is not None
        rd, rv, rb = res["run_draft"], res["run_verify"], res["run_batch"]

        def cd(*da, **dkw):
            stats["draft_steps"] += da[4]
            return rd(*da, **dkw)

        def cv(*va, **vkw):
            stats["verify_steps"] += 1
            return rv(*va, **vkw)

        res["run_draft"] = cd
        res["run_verify"] = cv
        if rb is not None:
            def cb(n_steps, nt_buf, h_buf, cur_tree, depth, *ba, **bkw):
                stats["verify_steps"] += n_steps
                stats["draft_steps"] += n_steps * depth
                return rb(n_steps, nt_buf, h_buf, cur_tree, depth, *ba, **bkw)
            res["run_batch"] = cb
        return res

    hpu_qwen35.build_mtp_runner = wrapped_builder

    clip = comfy.sd.load_clip([str(args.model)], clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
    tokens = clip.tokenize(settings["prompt"], system_prompt=sys_prompt, min_length=1, thinking=args.thinking)
    clip.load_model(tokens)

    mode_val = False if args.mode == "off" else (True if args.mode == "auto" else args.mode)
    runs = []
    telemetry_samples = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for turn in range(args.runs):
            stats["verify_steps"] = 0
            stats["draft_steps"] = 0
            stop_ev = threading.Event()
            th = None
            if turn == args.runs - 1 and args.telemetry:
                th = threading.Thread(target=sample_hl_smi, args=(stop_ev, telemetry_samples, 0.25), daemon=True)
                th.start()
            gc.collect()
            torch.hpu.synchronize()
            t0 = time.perf_counter()
            ids = clip.generate(
                tokens, do_sample=True, max_length=args.max_length,
                temperature=1.0, top_k=20, top_p=0.95, min_p=0.0,
                repetition_penalty=1.0, presence_penalty=1.5,
                seed=settings["sampling_mode.seed"], mtp=mode_val,
            )
            torch.hpu.synchronize()
            dt = time.perf_counter() - t0
            if th is not None:
                stop_ev.set()
                th.join(timeout=2.0)
            vsteps = stats["verify_steps"]
            dsteps = stats["draft_steps"]
            rec = {
                "turn": turn, "cold": turn == 0,
                "seconds": round(dt, 4), "tokens": len(ids),
                "tok_s": round(len(ids) / dt, 2),
                "verify_steps": vsteps, "draft_steps": dsteps,
                "acc_per_verify": round((len(ids) - 1) / max(1, vsteps), 4) if vsteps else 1.0,
                "drafts_per_verify": round(dsteps / max(1, vsteps), 3) if vsteps else 0.0,
                "ms_per_verify": round(dt * 1000.0 / max(1, vsteps), 3) if vsteps else round(dt * 1000.0 / len(ids), 3),
                "ms_per_token": round(dt * 1000.0 / len(ids), 3),
            }
            runs.append(rec)
            print("RUN:", rec, flush=True)
            if turn == args.runs - 1:
                (args.out_dir / f"caption-{args.label}.txt").write_text(clip.decode(ids))

    mem = torch.hpu.memory_stats()
    warm = [r for r in runs if not r["cold"]]
    tel_summary = {}
    if telemetry_samples:
        pw = [s["power_w"] for s in telemetry_samples if s["power_w"] is not None]
        tp = [s["temp_c"] for s in telemetry_samples if s["temp_c"] is not None]
        ut = [s["util_pct"] for s in telemetry_samples if s["util_pct"] is not None]
        ck = [s["clock_mhz"] for s in telemetry_samples if s["clock_mhz"] is not None]
        tel_summary = {
            "samples": len(telemetry_samples),
            "power_mean_w": round(statistics.mean(pw), 1) if pw else None,
            "power_max_w": round(max(pw), 1) if pw else None,
            "temp_max_c": round(max(tp), 1) if tp else None,
            "util_mean_pct": round(statistics.mean(ut), 1) if ut else None,
            "clock_min_mhz": round(min(ck), 1) if ck else None,
            "clock_max_mhz": round(max(ck), 1) if ck else None,
            "raw": telemetry_samples,
        }
    out = {
        "label": args.label, "model": str(args.model), "mode": args.mode,
        "thinking": args.thinking, "max_length": args.max_length,
        "shared_quant": not args.no_shared_quant, "use_fused_sdpa": not args.no_fused_sdpa,
        "kv_bucket": args.kv_bucket, "stage4": args.stage4,
        "runner_info": stats["runner_info"],
        "cold_seconds": runs[0]["seconds"],
        "warm_median_seconds": round(statistics.median(r["seconds"] for r in warm), 4) if warm else None,
        "warm_median_tok_s": round(statistics.median(r["tok_s"] for r in warm), 2) if warm else None,
        "warm_median_acc_per_verify": round(statistics.median(r["acc_per_verify"] for r in warm), 4) if warm else None,
        "runs": runs,
        "in_use_gib": round(mem["InUse"] / 1024**3, 3),
        "max_in_use_gib": round(mem["MaxInUse"] / 1024**3, 3),
        "alias_warnings": len([w for w in caught if "batch_as_strided" in str(w.message)]),
        "recompile_warnings": len([w for w in caught if "recompile_limit" in str(w.message)]),
        "telemetry": tel_summary,
    }
    (args.out_dir / f"full-{args.label}.json").write_text(json.dumps(out, indent=2))
    print("SUMMARY:", {k: v for k, v in out.items() if k not in ("runs", "telemetry")}, flush=True)
    if tel_summary:
        print("TELEMETRY:", {k: v for k, v in tel_summary.items() if k != "raw"}, flush=True)


def run_native_window(args):
    from moss_native_profile import NativeProfile

    wf = json.loads(Path(args.workflow).read_text())
    settings = wf["459:471"]["inputs"]
    sys_prompt = wf["459:475"]["inputs"]["value"]

    orig_builder = hpu_qwen35.build_mtp_runner
    state = {
        "mode": "warm",
        "step_idx": 0,
        "active": None,
        "win_t0": 0.0,
        "win_s": None,
        "win_steps_done": 0,
        "win_acc_toks": 0,
        "draft_in_win": 0,
        "in_window": False,
    }
    start_step = args.start_step
    win_steps = args.window_steps

    def wrapped_builder(*bargs, **bkwargs):
        bkwargs["shared_quant"] = not args.no_shared_quant
        bkwargs["use_fused_sdpa"] = not args.no_fused_sdpa
        bkwargs["kv_bucket"] = args.kv_bucket
        bkwargs["fuse_stage4"] = args.stage4
        if args.no_tpc:
            bkwargs["use_tpc"] = False
        if args.batch_steps is not None:
            bkwargs["batch_steps"] = args.batch_steps
        res = orig_builder(*bargs, **bkwargs)
        if res is None:
            return None
        rb = res["run_batch"]
        assert rb is not None, "Native window expects batched runner"

        def hooked_batch(n_steps, nt_buf, h_buf, cur_tree, depth, *ba, **bkw):
            s = state["step_idx"]
            # Align batch start/end to [start_step, start_step + win_steps)
            if not state["in_window"] and s >= start_step and state["win_s"] is None:
                state["in_window"] = True
                state["win_steps_done"] = 0
                state["win_acc_toks"] = 0
                state["draft_in_win"] = 0
                torch.hpu.synchronize()
                if state["mode"] == "profile":
                    state["active"] = NativeProfile(args.out_dir)
                    state["active"].__enter__()
                state["win_t0"] = time.perf_counter()
            out = rb(n_steps, nt_buf, h_buf, cur_tree, depth, *ba, **bkw)
            state["step_idx"] += n_steps
            if state["in_window"]:
                state["win_steps_done"] += n_steps
                state["draft_in_win"] += n_steps * depth
                state["win_acc_toks"] += sum(len(c) for c, _ in out)
                if state["win_steps_done"] >= win_steps:
                    torch.hpu.synchronize()
                    state["win_s"] = time.perf_counter() - state["win_t0"]
                    state["in_window"] = False
                    if state["active"] is not None:
                        state["active"].__exit__(None, None, None)
                        state["active"] = None
            return out

        res["run_batch"] = hooked_batch
        return res

    hpu_qwen35.build_mtp_runner = wrapped_builder

    clip = comfy.sd.load_clip([str(args.model)], clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
    tokens = clip.tokenize(settings["prompt"], system_prompt=sys_prompt, min_length=1, thinking=args.thinking)
    clip.load_model(tokens)
    mode_val = True if args.mode == "auto" else args.mode

    gen_len = max(args.max_length, (start_step + win_steps + 20) * 4)
    timings = []
    for run_mode in ("warm", "plain", "profile", "plain"):
        state["mode"] = run_mode
        state["step_idx"] = 0
        state["win_s"] = None
        state["in_window"] = False
        gc.collect()
        gc.disable()
        torch.hpu.synchronize()
        t0 = time.perf_counter()
        ids = clip.generate(
            tokens, do_sample=True, max_length=gen_len,
            temperature=1.0, top_k=20, top_p=0.95, min_p=0.0,
            repetition_penalty=1.0, presence_penalty=1.5,
            seed=settings["sampling_mode.seed"], mtp=mode_val,
        )
        torch.hpu.synchronize()
        dt = time.perf_counter() - t0
        gc.enable()
        rec = {
            "mode": run_mode, "total_s": round(dt, 4), "tokens": len(ids),
            "verify_steps_total": state["step_idx"],
            "window_s": round(state["win_s"], 6) if state["win_s"] else None,
            "window_verify_steps": state["win_steps_done"],
            "window_draft_steps": state["draft_in_win"],
            "window_acc_toks": state["win_acc_toks"],
        }
        timings.append(rec)
        print("NATIVE_RUN:", rec, flush=True)

    plain_vals = [r["window_s"] for r in timings if r["mode"] == "plain" and r["window_s"]]
    prof_rec = next(r for r in timings if r["mode"] == "profile")
    plain_med = statistics.median(plain_vals)
    exact_acc = prof_rec["window_acc_toks"]
    actual_vsteps = prof_rec["window_verify_steps"]
    prof_json = Path(os.environ["HABANA_PROF_CONFIG"])
    events_jsonl = args.out_dir / "native-events.jsonl"
    bmon = summarize_bmon(prof_json, events_jsonl, prof_rec["window_s"], plain_med, actual_vsteps, exact_acc)
    tl = summarize_native_timeline(events_jsonl, actual_vsteps, exact_acc)
    summary = {"label": args.label, "mode": args.mode, "timings": timings, "bmon": bmon, "timeline": tl}
    (args.out_dir / f"native-audit-{args.label}.json").write_text(json.dumps(summary, indent=2))
    print("BMON:", json.dumps(bmon, indent=2), flush=True)
    print("TIMELINE:", json.dumps(tl, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", type=Path)
    p.add_argument("--part", choices=["full", "native"], required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--workflow", default="/home/sonny/gaudi2/comfyui-hpu/workflows/qwen21-latest-2026-11-16-api.json")
    p.add_argument("--label", default="m3")
    p.add_argument("--mode", default="auto")
    p.add_argument("--thinking", action="store_true", default=True)
    p.add_argument("--no-thinking", dest="thinking", action="store_false")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--no-shared-quant", action="store_true")
    p.add_argument("--no-fused-sdpa", action="store_true")
    p.add_argument("--kv-bucket", type=int, default=0)
    p.add_argument("--stage4", action="store_true")
    p.add_argument("--no-tpc", action="store_true")
    p.add_argument("--batch-steps", type=int, default=None)
    p.add_argument("--telemetry", action="store_true")
    p.add_argument("--start-step", type=int, default=32)
    p.add_argument("--window-steps", type=int, default=24)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(32)
    if args.part == "full":
        run_full_bench(args)
    else:
        run_native_window(args)


if __name__ == "__main__":
    main()
