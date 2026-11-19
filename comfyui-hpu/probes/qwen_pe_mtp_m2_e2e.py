"""End-to-end ComfyUI HTTP validation & interleaved A/B benchmark for Qwen Image 2.1 FP8 PE+MTP."""
import argparse
import copy
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


def request_json(url: str, path: str, payload=None, timeout: int = 120):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else None


def extract_stage_timings(history_entry: dict):
    """Extract total and per-node wall-clock durations from ComfyUI history messages."""
    msgs = history_entry["status"]["messages"]
    start_ts = None
    end_ts = None
    cached = []
    node_ms = {}
    last_node = None
    last_ts = None
    for kind, data in msgs:
        ts = data.get("timestamp")
        if kind == "execution_start":
            start_ts = ts
            last_ts = ts
        elif kind == "execution_cached":
            cached = data.get("nodes", [])
        elif kind == "executing":
            nid = data.get("node")
            if last_node is not None and last_ts is not None and ts is not None:
                node_ms[last_node] = node_ms.get(last_node, 0.0) + (ts - last_ts)
            last_node = nid
            last_ts = ts
        elif kind == "execution_success":
            end_ts = ts
            if last_node is not None and last_ts is not None and ts is not None:
                node_ms[last_node] = node_ms.get(last_node, 0.0) + (ts - last_ts)
            last_node = None
    total_s = (end_ts - start_ts) / 1000.0 if (start_ts is not None and end_ts is not None) else None
    return total_s, {k: round(v / 1000.0, 4) for k, v in node_ms.items()}, cached


def run_prompt(url: str, graph: dict, product_dir: Path, out_dir: Path, label: str,
               prefix_base: str = "gaudi/qwen_pe_mtp_m2", timeout_s: int = 1200,
               require_compute: bool = True, require_loaders_cached: bool = False):
    g = copy.deepcopy(graph)
    g["461"]["inputs"]["filename_prefix"] = f"{prefix_base}/{label}"
    t0 = time.perf_counter()
    resp = request_json(url, "/prompt", {"prompt": g})
    pid = resp["prompt_id"]
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        hist = request_json(url, f"/history/{pid}")
        if pid in hist:
            break
        time.sleep(0.5)
    else:
        raise TimeoutError(f"Prompt {label} ({pid}) timed out")
    wall_s = time.perf_counter() - t0
    res = hist[pid]
    (out_dir / f"{label}-history.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    assert res["status"]["status_str"] == "success", f"{label} failed: {res['status']}"
    total_s, node_s, cached = extract_stage_timings(res)
    if require_compute:
        assert not ({"459:471", "459:452", "459:486", "459:457"} & set(cached)), (
            f"{label}: expected full compute, got cached={cached}"
        )
    if require_loaders_cached:
        expected_loaders = {"459:451", "459:453", "459:454", "459:473"}
        assert expected_loaders.issubset(set(cached)), (
            f"{label}: expected loaders cached, got cached={cached}"
        )
    caption = res["outputs"]["459:472"]["text"][0]
    (out_dir / f"{label}-caption.txt").write_text(caption)
    item = res["outputs"]["461"]["images"][0]
    img_path = product_dir / "output" / item["subfolder"] / item["filename"]
    assert img_path.exists(), f"Missing output image: {img_path}"
    img_sha = hashlib.sha256(img_path.read_bytes()).hexdigest()
    w, h = Image.open(img_path).size
    row = {
        "label": label,
        "prompt_id": pid,
        "wall_s": round(wall_s, 4),
        "exec_s": round(total_s, 4) if total_s is not None else None,
        "pe_s": node_s.get("459:471"),
        "clip_s": node_s.get("459:452"),
        "dit_s": node_s.get("459:486"),
        "vae_s": node_s.get("459:457"),
        "save_s": node_s.get("461"),
        "node_s": node_s,
        "cached_nodes": cached,
        "caption_chars": len(caption),
        "caption_sha256": hashlib.sha256(caption.encode()).hexdigest()[:16],
        "image_path": str(img_path),
        "image_size": [w, h],
        "image_sha256": img_sha[:16],
    }
    print("E2E:", json.dumps(row, ensure_ascii=False), flush=True)
    return row


def make_variant(base_graph: dict, pe_model: str, mtp_mode: str, *,
                 prompt: str = None, seed_delta: int = 0, thinking: bool = None,
                 aspect: str = None, megapixels: float = None):
    g = copy.deepcopy(base_graph)
    g["459:473"]["inputs"]["clip_name"] = pe_model
    g["459:471"]["inputs"]["mtp"] = mtp_mode
    if prompt is not None:
        g["459:471"]["inputs"]["prompt"] = prompt
        g["459:474"]["inputs"]["on_false"] = prompt
    if seed_delta:
        g["459:471"]["inputs"]["sampling_mode.seed"] += seed_delta
        g["459:488"]["inputs"]["noise_seed"] += seed_delta
    if thinking is not None:
        g["459:471"]["inputs"]["thinking"] = thinking
    if aspect is not None:
        g["13"]["inputs"]["aspect_ratio"] = aspect
    if megapixels is not None:
        g["13"]["inputs"]["megapixels"] = megapixels
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", type=Path)
    p.add_argument("--url", default="http://192.168.10.21:8189")
    p.add_argument("--product", type=Path, default=Path("/home/sonny/ComfyUI-Gaudi"))
    p.add_argument("--workflow", type=Path, default=Path("/home/sonny/gaudi2/comfyui-hpu/workflows/qwen21-latest-2026-11-16-api.json"))
    p.add_argument("--baseline-pe", default="qwen3.5_9b_qwen_image_2.1_pe_t2i.fp8_from_int8_fp32.safetensors")
    p.add_argument("--mtp-pe", required=True)
    p.add_argument("--mtp-mode", default="auto")
    p.add_argument("--warm-runs", type=int, default=5)
    p.add_argument("--prefix-base", default="gaudi/qwen_pe_mtp_m2")
    p.add_argument("--skip-baseline", action="store_true")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_wf = json.loads(args.workflow.read_text())
    base_prompt = base_wf["459:471"]["inputs"]["prompt"]

    q = request_json(args.url, "/queue")
    assert not q["queue_running"] and not q["queue_pending"], f"Queue not empty: {q}"

    def run_p(g, label, **kw):
        return run_prompt(args.url, g, args.product, args.out_dir, label, prefix_base=args.prefix_base, **kw)

    results = {
        "cold_mtp": None,
        "cold_off": None,
        "warm_mtp": [],
        "warm_off": [],
        "warm_tree3s1": [],
        "matrix": [],
        "baseline_nonmtp": [],
        "unload": [],
        "reload": None,
    }

    # 1. Cold run with MTP PE + target mode (loads UNET, CLIP, VAE, PE+MTP and compiles MTP graphs)
    g_mtp = make_variant(base_wf, args.mtp_pe, args.mtp_mode)
    results["cold_mtp"] = run_p(g_mtp, f"cold-{args.mtp_mode}")

    # 2. Warm-up single-token mtp="off" path on the exact same resident PE+MTP checkpoint (compiles single-token decode once)
    g_off = make_variant(base_wf, args.mtp_pe, "off")
    results["cold_off"] = run_p(
        g_off, "cold-off",
        require_loaders_cached=True,
    )

    # 3. True interleaved 5x warm A/B (mtp_mode vs off) with ALL 4 loaders kept 100% cached & ALL compute nodes re-executed
    for i in range(args.warm_runs):
        r_mtp = run_p(
            g_mtp, f"warm-{args.mtp_mode}-{i}",
            require_loaders_cached=True,
        )
        results["warm_mtp"].append(r_mtp)
        r_off = run_p(
            g_off, f"warm-off-{i}",
            require_loaders_cached=True,
        )
        results["warm_off"].append(r_off)

    # 4. Also verify fixed tree3s1 warm timing on the same prompt (alternating with auto keeps loaders cached)
    if args.mtp_mode != "tree3s1":
        g_t3 = make_variant(base_wf, args.mtp_pe, "tree3s1")
        results["warm_tree3s1"].append(run_p(
            g_t3, "tree3s1-0",
            require_loaders_cached=True,
        ))
        # One more after bouncing via auto to measure warm tree3s1 (after 3-step draft compiles once)
        results["matrix"].append(run_p(
            g_mtp, f"bounce-{args.mtp_mode}",
            require_loaders_cached=True,
        ))
        results["warm_tree3s1"].append(run_p(
            g_t3, "tree3s1-1",
            require_loaders_cached=True,
        ))

    # 5. Matrix: deterministic fixed-tree blue -> red -> blue-return isolation, auto prompt edit, seed+1, thinking=False, 512^2, 1024^2
    prompt_blue = base_prompt.replace("灰褐色", "深蓝色")
    prompt_red = base_prompt.replace("灰褐色", "深红色")
    cases = [
        ("blue-tree3s1", make_variant(base_wf, args.mtp_pe, "tree3s1", prompt=prompt_blue)),
        ("red-tree3s1", make_variant(base_wf, args.mtp_pe, "tree3s1", prompt=prompt_red)),
        ("blue-return-tree3s1", make_variant(base_wf, args.mtp_pe, "tree3s1", prompt=prompt_blue)),
        ("blue-auto", make_variant(base_wf, args.mtp_pe, args.mtp_mode, prompt=prompt_blue)),
        ("red-auto", make_variant(base_wf, args.mtp_pe, args.mtp_mode, prompt=prompt_red)),
        ("seed-plus1", make_variant(base_wf, args.mtp_pe, args.mtp_mode, seed_delta=1)),
        ("thinking-off", make_variant(base_wf, args.mtp_pe, args.mtp_mode, thinking=False)),
        ("sq512", make_variant(base_wf, args.mtp_pe, args.mtp_mode, aspect="1:1 (Square)", megapixels=0.262144)),
        ("sq1024", make_variant(base_wf, args.mtp_pe, args.mtp_mode, seed_delta=2, aspect="1:1 (Square)", megapixels=1.048576)),
    ]
    for cname, cg in cases:
        row = run_p(cg, cname, require_loaders_cached=True)
        results["matrix"].append(row)

    # Verify blue-tree3s1 == blue-return-tree3s1 EXACT caption AND pixel RGBA isolation!
    blue_cap = (args.out_dir / "blue-tree3s1-caption.txt").read_text()
    red_cap = (args.out_dir / "red-tree3s1-caption.txt").read_text()
    blue_ret_cap = (args.out_dir / "blue-return-tree3s1-caption.txt").read_text()
    row_by_label = {r["label"]: r for r in results["matrix"]}
    blue_rgba = np.asarray(Image.open(row_by_label["blue-tree3s1"]["image_path"]).convert("RGBA"))
    blue_ret_rgba = np.asarray(Image.open(row_by_label["blue-return-tree3s1"]["image_path"]).convert("RGBA"))
    red_rgba = np.asarray(Image.open(row_by_label["red-tree3s1"]["image_path"]).convert("RGBA"))
    results["blue_red_distinct"] = (blue_cap != red_cap) and (not np.array_equal(blue_rgba, red_rgba))
    results["blue_return_exact_caption"] = (blue_cap == blue_ret_cap)
    results["blue_return_exact_rgba"] = bool(np.array_equal(blue_rgba, blue_ret_rgba))
    assert results["blue_red_distinct"], "blue and red outputs were not distinct!"
    assert results["blue_return_exact_caption"], "blue-return caption did not match blue!"
    assert results["blue_return_exact_rgba"], "blue-return RGBA did not match blue!"

    # 6. Original non-MTP PE compatibility check (must still load cleanly and match warm_off!)
    if not args.skip_baseline:
        g_base_auto = make_variant(base_wf, args.baseline_pe, "auto")
        g_base_off = make_variant(base_wf, args.baseline_pe, "off")
        r_b0 = run_p(g_base_auto, "baseline-nonmtp-auto")
        r_b1 = run_p(g_base_off, "baseline-nonmtp-off", require_loaders_cached=True)
        results["baseline_nonmtp"].extend([r_b0, r_b1])
        # Verify exact match with warm_off on MTP checkpoint (non-MTP weights are byte-for-byte identical!)
        off_cap = (args.out_dir / "warm-off-0-caption.txt").read_text()
        b0_cap = (args.out_dir / "baseline-nonmtp-auto-caption.txt").read_text()
        off_rgba = np.asarray(Image.open(results["warm_off"][0]["image_path"]).convert("RGBA"))
        b0_rgba = np.asarray(Image.open(r_b0["image_path"]).convert("RGBA"))
        results["mtp_off_matches_baseline_caption"] = (off_cap == b0_cap)
        results["mtp_off_matches_baseline_rgba"] = bool(np.array_equal(off_rgba, b0_rgba))

    # 7. Unload (/free + 60s stability) + Reload check on MTP PE (leaves 8189 warm with MTP PE!)
    request_json(args.url, "/free", {"unload_models": True, "free_memory": True})
    for _ in range(6):
        time.sleep(10)
        results["unload"].append(request_json(args.url, "/system_stats"))
    results["reload"] = run_p(g_mtp, f"reload-{args.mtp_mode}")

    # Summarize
    warm_mtp_exec = [r["exec_s"] for r in results["warm_mtp"] if r["exec_s"] is not None]
    warm_mtp_pe = [r["pe_s"] for r in results["warm_mtp"] if r["pe_s"] is not None]
    warm_off_exec = [r["exec_s"] for r in results["warm_off"] if r["exec_s"] is not None]
    warm_off_pe = [r["pe_s"] for r in results["warm_off"] if r["pe_s"] is not None]
    results["summary"] = {
        "cold_mtp_exec_s": results["cold_mtp"]["exec_s"],
        "cold_mtp_pe_s": results["cold_mtp"]["pe_s"],
        "warm_mtp_exec_median_s": round(statistics.median(warm_mtp_exec), 4) if warm_mtp_exec else None,
        "warm_mtp_exec_range_s": [min(warm_mtp_exec), max(warm_mtp_exec)] if warm_mtp_exec else None,
        "warm_mtp_pe_median_s": round(statistics.median(warm_mtp_pe), 4) if warm_mtp_pe else None,
        "warm_mtp_pe_range_s": [min(warm_mtp_pe), max(warm_mtp_pe)] if warm_mtp_pe else None,
        "warm_off_exec_median_s": round(statistics.median(warm_off_exec), 4) if warm_off_exec else None,
        "warm_off_exec_range_s": [min(warm_off_exec), max(warm_off_exec)] if warm_off_exec else None,
        "warm_off_pe_median_s": round(statistics.median(warm_off_pe), 4) if warm_off_pe else None,
        "warm_off_pe_range_s": [min(warm_off_pe), max(warm_off_pe)] if warm_off_pe else None,
        "warm_tree3s1_exec_s": results["warm_tree3s1"][-1]["exec_s"] if results["warm_tree3s1"] else None,
        "warm_tree3s1_pe_s": results["warm_tree3s1"][-1]["pe_s"] if results["warm_tree3s1"] else None,
        "reload_exec_s": results["reload"]["exec_s"],
        "reload_pe_s": results["reload"]["pe_s"],
        "blue_red_distinct": results["blue_red_distinct"],
        "blue_return_exact_caption": results["blue_return_exact_caption"],
        "blue_return_exact_rgba": results["blue_return_exact_rgba"],
        "mtp_off_matches_baseline_caption": results.get("mtp_off_matches_baseline_caption"),
        "mtp_off_matches_baseline_rgba": results.get("mtp_off_matches_baseline_rgba"),
    }
    (args.out_dir / "e2e-summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("E2E_SUMMARY:", json.dumps(results["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
