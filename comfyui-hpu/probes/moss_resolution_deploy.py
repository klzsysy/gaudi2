"""Recheck the original prompt-expansion workflow on the ordinary service."""
import copy
import json
from pathlib import Path
import sys
import time
import urllib.request


root = Path(sys.argv[1])
url = sys.argv[2]
original = json.loads((root.parent / "performance-2026-11-12-resolution/original-history.json").read_text())
source = list(original.values())[-1]["prompt"][2]
records = []


def api(route, data=None):
    request = urllib.request.Request(url + route, data=None if data is None else json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


for trial in range(3):
    assert not any(api("/queue").values())
    graph = copy.deepcopy(source)
    graph["18:7"]["inputs"]["seed"] += trial
    prompt_id = api("/prompt", {"prompt": graph})["prompt_id"]
    start = time.monotonic()
    while prompt_id not in (history := api("/history/" + prompt_id)):
        if time.monotonic() - start > 1200:
            raise TimeoutError(prompt_id)
        time.sleep(.25)
    result = history[prompt_id]
    assert result["status"]["status_str"] == "success", result["status"]
    (root / f"deployment-{trial}-history.json").write_text(json.dumps(result, indent=2))
    status = dict(result["status"]["messages"])
    cached = status.get("execution_cached", {}).get("nodes", [])
    if trial:
        assert {"18:11", "18:4", "18:5"} <= set(cached), cached
    record = dict(trial=trial, prompt_id=prompt_id, cached_nodes=cached,
                  seconds=(status["execution_success"]["timestamp"] - status["execution_start"]["timestamp"]) / 1000)
    records.append(record)
    (root / "deployment.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(record), flush=True)
(root / "deployment-final-queue.json").write_text(json.dumps(api("/queue"), indent=2))
(root / "deployment-final-system.json").write_text(json.dumps(api("/system_stats"), indent=2))
