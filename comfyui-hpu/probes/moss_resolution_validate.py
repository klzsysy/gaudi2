"""Compare the cutoff repair with its prior path and check BF16 resolutions."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.parse
import urllib.request


root = Path(sys.argv[1])
source_history = json.loads((root.parent / "performance-2026-11-12-resolution/original-history.json").read_text())
source = list(source_history.values())[-1]
graph = copy.deepcopy(source["prompt"][2])
graph["18:4"]["inputs"]["text"] = source["outputs"]["18:13"]["text"][0]
graph.pop("18:13")
url = "http://127.0.0.1:8190"
records = []


def api(route, data=None):
    request = urllib.request.Request(url + route, data=None if data is None else json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read()
        return json.loads(body) if body else None


def run(label, width, height, seed, legacy=False, capture=False, bf16=False):
    prompt = copy.deepcopy(graph)
    prompt["18:6"]["inputs"].update(width=width, height=height)
    prompt["18:7"]["inputs"]["seed"] = seed
    if bf16:
        prompt["18:1"]["inputs"]["unet_name"] = "moss_image_sft_bf16.safetensors"
    extra = dict(moss_recompute=True, moss_recompute_nodes=["KSampler", "VAEDecode"],
                 resolution_legacy=legacy, resolution_capture=capture)
    prompt_id = api("/prompt", {"prompt": prompt, "extra_data": extra})["prompt_id"]
    start = time.monotonic()
    while prompt_id not in (history := api("/history/" + prompt_id)):
        if time.monotonic() - start > 1200:
            raise TimeoutError(prompt_id)
        time.sleep(.25)
    result = history[prompt_id]
    (root / f"{label}-history.json").write_text(json.dumps(result, indent=2))
    assert result["status"]["status_str"] == "success", result["status"]
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    events = [e for e in events if e.get("prompt_id") == prompt_id]
    end = next(e for e in events if e["kind"] == "request_end")
    assert not end["fallback_counts"], end["fallback_counts"]
    status = dict(result["status"]["messages"])
    times = {}
    for event in events:
        if event["kind"] == "node_start":
            times.setdefault(event["class_type"], []).append(-event["mono_ns"] / 1e9)
        elif event["kind"] == "node_end":
            times[event["class_type"]][-1] += event["mono_ns"] / 1e9
    record = dict(label=label, prompt_id=prompt_id, width=width, height=height, seed=seed, legacy=legacy,
                  bf16=bf16, seconds=(status["execution_success"]["timestamp"] - status["execution_start"]["timestamp"]) / 1000,
                  node_seconds={k: sum(v) for k, v in times.items()}, memory_stats=end["memory_stats"],
                  fallback_counts=end["fallback_counts"], cached_nodes=status.get("execution_cached", {}).get("nodes", []))
    records.append(record)
    (root / "validation.json").write_text(json.dumps(records, indent=2))
    for output in result["outputs"].values():
        for item in output.get("images", []):
            query = {k: item[k] for k in ("filename", "subfolder", "type")}
            with urllib.request.urlopen(url + "/view?" + urllib.parse.urlencode(query)) as response:
                (root / f"{label}.png").write_bytes(response.read())
    print(json.dumps({k: v for k, v in record.items() if k != "memory_stats"}), flush=True)


env = dict(os.environ, PT_HPU_LAZY_MODE="0", GC_KERNEL_PATH="/usr/lib/habanalabs/libtpc_kernels.so",
           PYTHONPATH=str(Path.cwd()), MOSS_EVENTS=str(root / "events.jsonl"), MOSS_PIPELINE_DIR=str(root))
command = [sys.executable, str(Path(__file__).with_name("moss_resolution_server.py")),
           "--disable-dynamic-vram", "--disable-pinned-memory", "--disable-async-offload",
           "--disable-xformers", "--use-pytorch-cross-attention", "--disable-all-custom-nodes",
           "--listen", "127.0.0.1", "--port", "8190"]
with (root / "server.log").open("w") as log:
    server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            if server.poll() is not None:
                raise RuntimeError("Validation server exited")
            try:
                api("/system_stats")
                break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError("Validation server startup")
        for width, height in ((1024, 1824), (1088, 1936)):
            for trial in range(4):
                run(f"fixed-{width}x{height}-{trial}", width, height, 734670000 + max(trial - 1, 0))
        for trial in range(4):
            run(f"legacy-1088x1936-{trial}", 1088, 1936, 734670000 + max(trial - 1, 0), legacy=True)
        for height in (1936, 1952, 1968):
            for trial in range(2):
                run(f"fixed-boundary-{height}-{trial}", 1024, height, 734670000 + trial)
        run("legacy-capture", 1088, 1936, 734670000, legacy=True, capture=True)
        for height in (1936, 1952, 1968):
            for trial in range(2):
                run(f"bf16-boundary-{height}-{trial}", 1024, height, 734670000 + trial, bf16=True)
        release = root / "release"
        release.mkdir(exist_ok=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name("moss_memory_release.py")), str(release), url], check=True)
        run("fixed-reload-512", 512, 512, 734670000)
        assert not any(api("/queue").values())
    finally:
        server.terminate()
        server.wait(timeout=60)
