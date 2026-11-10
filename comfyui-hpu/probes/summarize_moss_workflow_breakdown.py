"""Reconcile one recorded warm request with its non-overlapping node intervals."""
import argparse
import collections
import json
from pathlib import Path
import statistics


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
parser.add_argument("--label", help="Default: the middle warm request ordered by duration")
args = parser.parse_args()
runs = [json.loads(line) for line in (args.directory / "runs.jsonl").read_text().splitlines()]
events = [json.loads(line) for line in (args.directory / "events.jsonl").read_text().splitlines()]
warm = [r for r in runs if "warm" in r["label"] and not r.get("diagnostic", False)]
ordered = sorted(warm, key=lambda r: r["request_seconds"])
selected = next(r for r in warm if r["label"] == args.label) if args.label else ordered[len(ordered) // 2]
rows = []
for run in warm:
    relevant = [e for e in events if e.get("prompt_id") == run["prompt_id"]]
    start = next(e for e in relevant if e["kind"] == "request_start")
    end = next(e for e in relevant if e["kind"] == "request_end")
    total_ns = end["mono_ns"] - start["mono_ns"]
    pending, nodes = {}, []
    for event in relevant:
        if event["kind"] == "node_start":
            assert event["node"] not in pending, "Repeated node start without an end"
            pending[event["node"]] = event
        elif event["kind"] == "node_end":
            opening = pending.pop(event["node"])
            nodes.append({"node": event["node"], "class_type": event["class_type"],
                          "start_ns": opening["mono_ns"] - start["mono_ns"],
                          "end_ns": event["mono_ns"] - start["mono_ns"]})
    assert not pending, "Unclosed node intervals"
    nodes.sort(key=lambda n: n["start_ns"])
    previous_end, covered_ns = 0, 0
    phases = collections.defaultdict(float)
    for node in nodes:
        assert previous_end <= node["start_ns"] <= node["end_ns"] <= total_ns, "Overlapping/out-of-request intervals"
        previous_end = node["end_ns"]
        duration_ns = node["end_ns"] - node["start_ns"]
        covered_ns += duration_ns
        node.update(seconds=duration_ns / 1e9, percent=duration_ns / total_ns * 100)
        phases[node["class_type"]] += node["seconds"]
    progress = [e for e in relevant if e["kind"] == "progress" and e.get("class_type") == "TextGenerate"]
    rows.append({"label": run["label"], "prompt_id": run["prompt_id"],
                 "seconds": total_ns / 1e9, "output_tokens": run.get("output_tokens"),
                 "fallback_counts": end.get("fallback_counts"),
                 "node_seconds_by_class": dict(phases), "nodes": nodes,
                 "outside_nodes_seconds": (total_ns - covered_ns) / 1e9,
                 "text_progress_updates": len(progress)})

print(json.dumps({
    "measurement": "Recorded host wall time; no per-node synchronization in non-diagnostic runs",
    "warm_count": len(warm), "warm_median_seconds": statistics.median(r["request_seconds"] for r in warm),
    "selected_request": next(r for r in rows if r["prompt_id"] == selected["prompt_id"]),
    "warm_requests": [{k: v for k, v in r.items() if k != "nodes"} for r in rows],
}, indent=2))
