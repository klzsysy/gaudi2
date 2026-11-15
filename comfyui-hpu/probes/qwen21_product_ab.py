"""Generate paired ordinary Qwen workflows from two verified product revisions."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image


ROOT = Path("/home/sonny/gaudi2/comfyui-hpu/quality-qwen21-product-ab-2026-11-15")
PRODUCT = Path("/home/sonny/ComfyUI-Gaudi")
BASE = ROOT.parent
URL = "http://127.0.0.1:8190"
VERSIONS = json.loads((ROOT / "versions.json").read_text())
WORKFLOW = json.loads((BASE / "workflows/qwen21-latest-2026-11-15-api.json").read_text())
REFERENCE = json.loads((BASE / "performance-qwen-hbm-2026-11-15/restored.json").read_text())
TEXT_SEED = WORKFLOW["459:471"]["inputs"]["sampling_mode.seed"]
IMAGE_SEED = WORKFLOW["459:488"]["inputs"]["noise_seed"]
EXTRA = REFERENCE["prompt"][3]["extra_pnginfo"]
server = None
log = None


def request(path, payload=None, url=URL):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as response:
        data = response.read()
    return json.loads(data) if data else None


def wait_idle(url):
    while True:
        queue = request("/queue", url=url)
        if not queue["queue_running"] and not queue["queue_pending"]:
            return
        time.sleep(2)


def verify_version(label):
    expected = VERSIONS[label]["files"]
    actual = {name: hashlib.sha256((PRODUCT / name).read_bytes()).hexdigest() for name in expected}
    assert actual == expected, (label, actual)
    return actual


def deploy(label):
    for name in VERSIONS[label]["files"]:
        target = PRODUCT / name
        temporary = target.with_name(target.name + ".ab-tmp")
        shutil.copyfile(ROOT / "sources" / label / name, temporary)
        os.replace(temporary, target)
    (PRODUCT / ".run/deployed-version").write_text(VERSIONS[label]["commit"] + "\n")
    return verify_version(label)


def stop_test_server():
    global server, log
    if server is not None:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        server = None
    if log is not None:
        log.close()
        log = None


def start_test_server(label):
    global server, log
    log = (ROOT / f"{label}-server.log").open("a")
    env = dict(os.environ, PT_HPU_LAZY_MODE="0", GC_KERNEL_PATH="/usr/lib/habanalabs/libtpc_kernels.so")
    server = subprocess.Popen(["bash", "scripts/gaudi/start.sh", "--listen", "127.0.0.1", "--port", "8190"],
                              cwd=PRODUCT, env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        if server.poll() is not None:
            raise RuntimeError(f"{label} server exited with {server.returncode}")
        try:
            request("/queue")
            return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise RuntimeError(f"{label} server failed to become ready")


def replace_seeds(value, offset):
    if isinstance(value, dict):
        return {key: replace_seeds(item, offset) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_seeds(item, offset) for item in value]
    if isinstance(value, int) and value in (TEXT_SEED, IMAGE_SEED):
        return value + offset
    return value


def verify_recomputed(result):
    required = {node for node, definition in WORKFLOW.items()
                if definition["class_type"] in ("TextGenerate", "SamplerCustomAdvanced")}
    cached = [data["nodes"] for kind, data in result["status"]["messages"] if kind == "execution_cached"]
    assert all(not required.intersection(nodes) for nodes in cached), cached
    return cached


def generate(label, index):
    folder = ROOT / "review" / label
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{index + 1:02d}"
    history_path = ROOT / f"{label}-{name}.json"
    if history_path.exists() and (folder / f"{name}.png").exists():
        result = json.loads(history_path.read_text())
        assert result["status"]["status_str"] == "success"
        verify_recomputed(result)
        print(json.dumps(dict(event="reused_verified", version=label, case=name)), flush=True)
        return
    graph = copy.deepcopy(WORKFLOW)
    graph["459:471"]["inputs"]["sampling_mode.seed"] = TEXT_SEED + index
    graph["459:488"]["inputs"]["noise_seed"] = IMAGE_SEED + index
    graph["461"]["inputs"]["filename_prefix"] = f"gaudi/qwen21_product_ab/{label}/{name}"
    extra = dict(extra_pnginfo=replace_seeds(EXTRA, index))
    pid = request("/prompt", dict(prompt=graph, extra_data=extra))["prompt_id"]
    (ROOT / "active.json").write_text(json.dumps(dict(version=label, case=name, prompt_id=pid), indent=2))
    print(json.dumps(dict(event="submitted", version=label, case=name, prompt_id=pid)), flush=True)
    while True:
        if server.poll() is not None:
            raise RuntimeError(f"{label} server exited during {name}")
        try:
            history = request("/history/" + pid)
        except TimeoutError:
            continue
        if pid in history:
            break
        time.sleep(2)
    result = history[pid]
    history_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    assert result["status"]["status_str"] == "success", result["status"]
    image = result["outputs"]["461"]["images"][0]
    source = PRODUCT / "output" / image["subfolder"] / image["filename"]
    shutil.copy2(source, folder / f"{name}.png")
    caption = result["outputs"]["459:472"]["text"][0]
    (folder / f"{name}.txt").write_text(caption)
    stamps = [data["timestamp"] for kind, data in result["status"]["messages"]
              if kind in ("execution_start", "execution_success")]
    cached = verify_recomputed(result)
    record = dict(version=label, commit=VERSIONS[label]["commit"], case=name, prompt_id=pid,
                  text_seed=TEXT_SEED + index, image_seed=IMAGE_SEED + index,
                  seconds=(stamps[-1] - stamps[0]) / 1000, cached_nodes=cached,
                  source=str(source), image=f"{label}/{name}.png", caption=f"{label}/{name}.txt")
    with (ROOT / "runs.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(dict(event="completed", **record)), flush=True)


def compare():
    rows = []
    for index in range(8):
        name = f"{index + 1:02d}"
        paths = [ROOT / "review" / version / f"{name}.png" for version in ("current", "previous")]
        arrays = [np.asarray(Image.open(path).convert("RGBA")) for path in paths]
        assert arrays[0].shape == arrays[1].shape
        difference = np.abs(arrays[0].astype(np.int16) - arrays[1].astype(np.int16))
        captions = [(ROOT / "review" / version / f"{name}.txt").read_text() for version in ("current", "previous")]
        records = {}
        for version in ("current", "previous"):
            result = json.loads((ROOT / f"{version}-{name}.json").read_text())
            verify_recomputed(result)
            graph = result["prompt"][2]
            assert graph["459:471"]["inputs"]["sampling_mode.seed"] == TEXT_SEED + index
            assert graph["459:488"]["inputs"]["noise_seed"] == IMAGE_SEED + index
            stamps = [data["timestamp"] for kind, data in result["status"]["messages"]
                      if kind in ("execution_start", "execution_success")]
            records[version] = dict(prompt_id=result["prompt"][1], seconds=(stamps[-1]-stamps[0])/1000)
        rows.append(dict(case=name, text_seed=TEXT_SEED + index, image_seed=IMAGE_SEED + index,
                         rgba_exact=bool(np.array_equal(*arrays)), changed_pixels=int(np.any(difference, axis=-1).sum()),
                         rgba_mae=float(difference.mean()), max_abs=int(difference.max()),
                         caption_exact=captions[0] == captions[1], width=arrays[0].shape[1], height=arrays[0].shape[0],
                         pixel_sha256=[hashlib.sha256(array.tobytes()).hexdigest() for array in arrays], runs=records))
    (ROOT / "comparison.json").write_text(json.dumps(rows, indent=2))
    (ROOT / "review/manifest.json").write_text(json.dumps(dict(versions=VERSIONS, prompt=WORKFLOW["459:471"]["inputs"]["prompt"],
                                                               workflow=WORKFLOW, pairs=rows), ensure_ascii=False, indent=2))
    print(json.dumps(dict(event="comparison", pairs=rows)), flush=True)


verify_version("current")
wait_idle("http://192.168.10.21:8189")
subprocess.run(["./stop-hpu.sh"], cwd=PRODUCT, check=True)
try:
    for label in ("current", "previous"):
        deployed = deploy(label)
        (ROOT / f"{label}-source-sha256.json").write_text(json.dumps(deployed, indent=2))
        start_test_server(label)
        for index in range(8):
            generate(label, index)
        wait_idle(URL)
        stop_test_server()
    compare()
finally:
    stop_test_server()
    restored = deploy("current")
    subprocess.run(["./start-hpu.sh"], cwd=PRODUCT, check=True, timeout=180)
    queue = request("/queue", url="http://192.168.10.21:8189")
    (ROOT / "restoration.json").write_text(json.dumps(dict(commit=VERSIONS["current"]["commit"], files=restored, queue=queue), indent=2))
