"""Verify the deployed ordinary API against the validated natural Qwen outputs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from PIL import Image


base = Path("/home/sonny/gaudi2/comfyui-hpu")
root = base / "deploy-qwen-execution-2026-11-15"
reference_root = base / "performance-qwen-execution-2026-11-15/workflow/gdn_direct-sdpa-whole"
product = Path("/home/sonny/ComfyUI-Gaudi")
workflow = json.loads((base / "workflows/qwen21-latest-2026-11-15-api.json").read_text())


def request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request("http://192.168.10.21:8189" + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as response:
        content = response.read()
    return json.loads(content) if content else None


def pixels(history):
    image = history["outputs"]["461"]["images"][0]
    path = product / "output" / image["subfolder"] / image["filename"]
    value = Image.open(path).convert("RGBA")
    return value.size, hashlib.sha256(value.tobytes()).hexdigest(), str(path)


parser = argparse.ArgumentParser()
parser.add_argument("--reload-only", action="store_true")
parser.add_argument("--warm-only", action="store_true")
args = parser.parse_args()
results = json.loads((root / "online-summary.json").read_text()) if args.reload_only or args.warm_only else []
cases = [(f"{index+1:02}", index) for index in range(2)]
if args.reload_only:
    cases = [("reload", 0)]
if args.warm_only:
    cases = [("final", 0)]
if args.reload_only:
    queue = request("/queue")
    assert not queue["queue_running"] and not queue["queue_pending"], queue
    request("/free", dict(unload_models=True, free_memory=True))
    samples = []
    for _ in range(4):
        time.sleep(10)
        samples.append(request("/system_stats")["devices"])
    (root / "online-free.json").write_text(json.dumps(samples, indent=2))
for label, index in cases:
    graph = copy.deepcopy(workflow)
    graph["459:471"]["inputs"]["sampling_mode.seed"] += index
    graph["459:488"]["inputs"]["noise_seed"] += index
    graph["461"]["inputs"]["filename_prefix"] = f"gaudi/qwen21_execution_deployed/{label}"
    pid = request("/prompt", dict(prompt=graph))["prompt_id"]
    print(dict(case=label, prompt_id=pid), flush=True)
    while True:
        history = request("/history/" + pid)
        if pid in history:
            break
        time.sleep(1)
    result = history[pid]
    (root / f"online-{label}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    assert result["status"]["status_str"] == "success", result["status"]
    reference = json.loads((reference_root / f"paired_seed_{index+1:02}-0.json").read_text())
    actual, expected = pixels(result), pixels(reference)
    caption_exact = result["outputs"]["459:472"] == reference["outputs"]["459:472"]
    timestamps = [data["timestamp"] for kind, data in result["status"]["messages"]
                  if kind in ("execution_start", "execution_success")]
    required = {key for key, node in graph.items()
                if node["class_type"] in ("TextGenerate", "SamplerCustomAdvanced")}
    for kind, data in result["status"]["messages"]:
        if kind == "execution_cached":
            assert not required.intersection(data["nodes"])
    record = dict(case=label, prompt_id=pid, seconds=(timestamps[-1] - timestamps[0]) / 1000,
                  caption_exact=caption_exact, rgba_exact=actual[:2] == expected[:2],
                  size=actual[0], output=actual[2], reference=expected[2])
    results.append(record)
    (root / "online-summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(record), flush=True)
    assert caption_exact and record["rgba_exact"], record

version = json.loads((root / "version.json").read_text())
actual_hashes = {name: hashlib.sha256((product / name).read_bytes()).hexdigest() for name in version["files"]}
assert actual_hashes == version["files"]
final = dict(commit=(product / ".run/deployed-version").read_text().strip(), files=actual_hashes,
             queue=request("/queue"), runs=results)
(root / "online-final.json").write_text(json.dumps(final, indent=2))
print(json.dumps(final), flush=True)
