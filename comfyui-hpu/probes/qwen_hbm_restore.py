"""Verify the ordinary service and original 2MP image after isolated experiments."""
import json
from pathlib import Path
import time
import urllib.request

from PIL import Image


root = Path("/home/sonny/gaudi2/comfyui-hpu/performance-qwen-hbm-2026-11-15")
base = root.parent
reference = json.loads((base / "performance-qwen21-text-next-2026-11-15/restored.json").read_text())
workflow = json.loads((base / "workflows/qwen21-latest-2026-11-15-api.json").read_text())
workflow["461"]["inputs"]["filename_prefix"] = "gaudi/qwen21_single_bf16/restored"


def request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request("http://192.168.10.21:8189" + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read())


queue = request("/queue")
assert not queue["queue_running"] and not queue["queue_pending"]
pid = request("/prompt", dict(prompt=workflow, extra_data=reference["prompt"][3]))["prompt_id"]
print(pid, flush=True)
while True:
    history = request("/history/" + pid)
    if pid in history:
        break
    time.sleep(1)
result = history[pid]
(root / "restored.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
assert result["status"]["status_str"] == "success"
stamps = [data["timestamp"] for kind, data in result["status"]["messages"]
          if kind in ("execution_start", "execution_success")]


def image_path(record):
    image = record["outputs"]["461"]["images"][0]
    return Path("/home/sonny/ComfyUI-Gaudi/output") / image["subfolder"] / image["filename"]


actual = Image.open(image_path(result)).convert("RGBA")
expected = Image.open(image_path(reference)).convert("RGBA")
equal = actual.size == expected.size and actual.tobytes() == expected.tobytes()
queue = request("/queue")
summary = dict(prompt_id=pid, seconds=(stamps[-1]-stamps[0])/1000,
               rgba_exact=equal, size=actual.size, output=str(image_path(result)),
               reference=str(image_path(reference)), queue=queue,
               deployed=Path("/home/sonny/ComfyUI-Gaudi/.run/deployed-version").read_text().strip())
(root / "restoration.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
assert equal and not queue["queue_running"] and not queue["queue_pending"]
