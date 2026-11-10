"""Serial B0/B1 replay client for moss_observed_server.py."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

directory = Path(sys.argv[1])
source = json.loads((directory / "b0-source.json").read_text())
url = "http://127.0.0.1:8190"


def api(path, data=None):
    req = urllib.request.Request(url + path, data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def run(label, graph, diagnostic=False):
    (directory / (label + "-api.json")).write_text(json.dumps(graph, indent=2))
    start = time.perf_counter()
    submitted = api("/prompt", {"prompt": graph, "extra_data": {"moss_diagnostic": diagnostic}})
    prompt_id = submitted["prompt_id"]
    while True:
        history = api("/history/" + prompt_id)
        if prompt_id in history:
            break
        if time.perf_counter() - start > 1800:
            raise TimeoutError(prompt_id)
        time.sleep(0.2)
    elapsed = time.perf_counter() - start
    history = history[prompt_id]
    (directory / (label + "-history.json")).write_text(json.dumps(history, indent=2))
    assert history["status"]["status_str"] == "success", history["status"]
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    events = [x for x in events if x.get("prompt_id") == prompt_id]
    executed = [x for x in events if x["kind"] == "node_end"]
    classes = [x["class_type"] for x in executed]
    for expected in ["KSampler", "VAEDecode", "CLIPTextEncode"] + ([] if label.startswith("b1") else ["TextGenerate"]):
        assert expected in classes, (label, expected, classes)
    images = []
    for output in history["outputs"].values():
        for metadata in output.get("images", []):
            with urllib.request.urlopen(url + "/view?" + urllib.parse.urlencode(metadata), timeout=30) as response:
                payload = response.read()
            image = Image.open(io.BytesIO(payload)).convert("RGB")
            pixels = np.asarray(image)
            filename = label + "-" + str(len(images)) + ".png"
            (directory / filename).write_bytes(payload)
            images.append({"file": filename, "size": image.size, "min": int(pixels.min()),
                           "max": int(pixels.max()), "mean": float(pixels.mean()), "std": float(pixels.std()),
                           "sha256": hashlib.sha256(payload).hexdigest()})
    assert images and all(x["std"] > 1 for x in images)
    record = {"label": label, "prompt_id": prompt_id, "client_seconds": elapsed,
              "images": images, "executed_classes": classes, "diagnostic": diagnostic}
    requests = [x for x in events if x["kind"] in {"request_start", "request_end"}]
    if len(requests) == 2:
        record["request_seconds"] = (requests[1]["mono_ns"] - requests[0]["mono_ns"]) / 1e9
        record["fallback_counts_cumulative"] = requests[1]["fallback_counts"]
    generation = [x for x in events if x["kind"] == "generate_end"]
    if generation:
        record["output_tokens"] = generation[0]["output_tokens"]
        record["caption"] = generation[0]["text"]
    with (directory / "runs.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps({k:v for k,v in record.items() if k not in {"caption", "images"}}), flush=True)
    return record


assert api("/queue") == {"queue_running": [], "queue_pending": []}
b0_runs = int(os.environ.get('MOSS_B0_RUNS', '6'))
for index in range(b0_runs):
    diagnostic = os.environ.get('MOSS_DIAGNOSTIC_LAST') == '1' and index == b0_runs - 1
    record = run("b0-" + ("cold" if index == 0 else f"warm-{index}"), source, diagnostic=diagnostic)
    if index == 0:
        caption = record["caption"]
        (directory / "fixed-caption.txt").write_text(caption)

b1 = copy.deepcopy(source)
b1["18:16"]["inputs"]["switch"] = False
b1["18:17"]["inputs"]["value"] = caption
b1_runs = int(os.environ.get('MOSS_B1_RUNS', '6'))
for index in range(b1_runs):
    run("b1-" + ("transition" if index == 0 else f"warm-{index}"), b1)
if b1_runs:
    run("b1-diagnostic", b1, diagnostic=True)
records = [json.loads(x) for x in (directory / "runs.jsonl").read_text().splitlines()]
summary = {}
for case in ["b0", "b1"]:
    values = [x["request_seconds"] for x in records if x["label"].startswith(case + "-warm") and not x["diagnostic"]]
    if not values:
        continue
    summary[case] = {"warm_samples": len(values), "median_seconds": statistics.median(values),
                     "min_seconds": min(values), "max_seconds": max(values),
                     "p10_seconds": float(np.percentile(values, 10)), "p90_seconds": float(np.percentile(values, 90))}
(directory / "baseline-summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
