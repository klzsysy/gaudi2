"""Join baseline events and low-rate HPU telemetry without inferring HBM bytes."""
import datetime
import json
from pathlib import Path
import statistics
import sys
from zoneinfo import ZoneInfo

p = Path(sys.argv[1])
runs = [json.loads(x) for x in (p / "runs.jsonl").read_text().splitlines()]
events = [json.loads(x) for x in (p / "events.jsonl").read_text().splitlines()]
telemetry = []
for line in (p / "hl-smi.log").read_text().splitlines():
    row = [x.strip() for x in line.split(",")]
    if len(row) != 8:
        continue
    stamp = datetime.datetime.strptime(row[0].replace(" CST", ""), "%a %b %d %H:%M:%S %Y")
    telemetry.append({"wall_ns": int(stamp.replace(tzinfo=ZoneInfo("Asia/Shanghai")).timestamp() * 1e9),
                      "utilization_aip": float(row[1]), "memory_used_MiB": float(row[3]),
                      "power_W": float(row[4]), "power_limit_W": float(row[5])/1000,
                      "temperature_C": float(row[6]), "clock_MHz": float(row[7])})

summary = []
for run in runs:
    selected = [x for x in events if x.get("prompt_id") == run["prompt_id"]]
    starts = {}
    segments = []
    for x in selected:
        if x["kind"] == "node_start":
            starts[x["node"]] = x
        elif x["kind"] == "node_end":
            start = starts.pop(x["node"])
            samples = [t for t in telemetry if start["wall_ns"] <= t["wall_ns"] <= x["wall_ns"]]
            segment = {"node": x["node"], "class_type": x["class_type"],
                       "seconds": (x["mono_ns"]-start["mono_ns"])/1e9,
                       "telemetry_samples": len(samples)}
            if samples:
                segment.update({key+"_mean": statistics.mean(t[key] for t in samples)
                                for key in ["utilization_aip", "power_W", "temperature_C", "clock_MHz"]})
                segment["power_W_max"] = max(t["power_W"] for t in samples)
            segments.append(segment)
    total = sum(x["seconds"] for x in segments)
    steps = [x for x in selected if x["kind"] == "progress" and x["class_type"] == "KSampler"]
    itl = [(b["mono_ns"]-a["mono_ns"])/1e9 for a,b in zip(steps,steps[1:])]
    generated = [x for x in selected if x["kind"] in {"generate_start", "generate_end"}]
    record = {"label": run["label"], "request_seconds": run["request_seconds"],
              "diagnostic_sync": run["diagnostic"], "segments": segments,
              "node_coverage_fraction": total/run["request_seconds"],
              "step_intervals_seconds": itl,
              "tensor_checks": [x for x in selected if x["kind"] == "tensor_check"]}
    if len(generated) == 2:
        seconds = (generated[1]["mono_ns"]-generated[0]["mono_ns"])/1e9
        record["generation"] = {"seconds_including_prefill": seconds,
                                 "output_tokens": generated[1]["output_tokens"],
                                 "tokens_per_second_including_prefill": generated[1]["output_tokens"]/seconds,
                                 "token_ids": generated[1]["token_ids"]}
    summary.append(record)
(p / "stage-summary.json").write_text(json.dumps(summary, indent=2))
for record in summary:
    print(record["label"], round(record["request_seconds"],3),
          "node coverage", round(record["node_coverage_fraction"],4),
          [(x["class_type"],round(x["seconds"],3)) for x in record["segments"]])
