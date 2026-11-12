"""Measure resolution changes through the running service and its normal cache."""
import argparse
import asyncio
import copy
import json
from pathlib import Path
import time
import uuid

import aiohttp


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8189")
    parser.add_argument("--cases", default="1024x1824,1088x1936")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--label", default="paired")
    args = parser.parse_args()
    history = json.loads((args.root / "original-history.json").read_text())
    source = list(history.values())[-1]["prompt"][2]
    records = []
    client_id = str(uuid.uuid4())
    async with aiohttp.ClientSession() as session:
        async def api(route, data=None):
            method = session.get if data is None else session.post
            kwargs = {} if data is None else {"json": data}
            async with method(args.url + route, **kwargs) as response:
                response.raise_for_status()
                return await response.json()

        async with session.ws_connect(args.url + "/ws?clientId=" + client_id) as ws:
            for trial in range(args.rounds):
                for size in args.cases.split(","):
                    width, height = map(int, size.split("x"))
                    queue = await api("/queue")
                    if any(queue.values()):
                        raise RuntimeError("Service queue is occupied")
                    graph = copy.deepcopy(source)
                    graph["18:6"]["inputs"].update(width=width, height=height)
                    graph["18:7"]["inputs"]["seed"] = 734670000 + trial
                    label = f"{args.label}-{trial}-{size}"
                    submitted = await api("/prompt", {"prompt": graph, "client_id": client_id})
                    prompt_id = submitted["prompt_id"]
                    events = []
                    started = time.monotonic()
                    while True:
                        message = await ws.receive(timeout=300)
                        if message.type != aiohttp.WSMsgType.TEXT:
                            if message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                                raise RuntimeError(str(message))
                            continue
                        event = json.loads(message.data)
                        if event.get("data", {}).get("prompt_id") != prompt_id:
                            continue
                        events.append(dict(received_s=time.monotonic() - started, **event))
                        if event["type"] in ("execution_success", "execution_error", "execution_interrupted"):
                            break
                    result = (await api("/history/" + prompt_id))[prompt_id]
                    (args.root / f"{label}.json").write_text(json.dumps(dict(history=result, events=events), indent=2))
                    assert result["status"]["status_str"] == "success", result["status"]
                    status = dict(result["status"]["messages"])
                    cached = status.get("execution_cached", {}).get("nodes", [])
                    assert {"18:11", "18:4", "18:5"}.issubset(cached), cached
                    total = (status["execution_success"]["timestamp"] - status["execution_start"]["timestamp"]) / 1000
                    transitions = [e for e in events if e["type"] == "executing"]
                    durations = {}
                    for current, following in zip(transitions, transitions[1:]):
                        node = current["data"]["node"]
                        if node is not None:
                            durations[graph[node]["class_type"]] = following["received_s"] - current["received_s"]
                    progress = [e["received_s"] for e in events if e["type"] == "progress" and e["data"].get("max") == 25]
                    record = dict(label=label, prompt_id=prompt_id, width=width, height=height,
                                  seed=graph["18:7"]["inputs"]["seed"], seconds=total,
                                  node_seconds=durations, progress_seconds=progress, cached_nodes=cached)
                    records.append(record)
                    (args.root / f"{args.label}-summary.json").write_text(json.dumps(records, indent=2))
                    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
