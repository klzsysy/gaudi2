"""Integrate raw BMON samples, preserving per-port and per-window diagnostics."""
import argparse
import collections
import json
from pathlib import Path
import statistics


def summarize(directory):
    ports = collections.defaultdict(list)
    beats = collections.defaultdict(list)
    for line in (directory / "native-events.jsonl").open():
        event = json.loads(line)
        if event["category"] == "BMON BW":
            ports[event["name"]].append(event)
        if event["category"] == "BMON" and "counter_values" in event:
            beats[event["name"]].append(event)
    if (directory / "run.json").exists():
        result = json.loads((directory / "run.json").read_text())
    else:
        result = dict(operation="text", **json.loads((directory / "window.json").read_text()))
        if (directory / "plain-window.json").exists():
            result["plain_window"] = json.loads((directory / "plain-window.json").read_text())
    result["ports"] = {}
    result["integrated_bytes"] = dict(read=0, write=0)
    for name, events in ports.items():
        events.sort(key=lambda e: e["ts"])
        periods = [b["ts"] - a["ts"] for a, b in zip(events, events[1:])]
        typical = statistics.median(periods)
        integral = sum(event["counter_values"][2] * period * 1e-6
                       for event, period in zip(events[1:], periods))
        constant = sum(event["counter_values"][2] for event in events) * typical * 1e-6
        result["ports"][name] = dict(count=len(events), period_us=typical,
                                    min_period_us=min(periods), max_period_us=max(periods),
                                    integrated_bytes=integral, constant_period_bytes=constant,
                                    avg_exceeds_max=sum(e["counter_values"][2] > e["counter_values"][1] for e in events))
        direction = "read" if "Read" in name else "write"
        result["integrated_bytes"][direction] += integral
    for direction in ("read", "write"):
        known = result.get(f"known_{direction}_bytes", 0)
        result[f"{direction}_ratio"] = result["integrated_bytes"][direction] / known if known else None
    if beats:
        configuration = json.loads((directory / "profiler.json").read_text())
        plugin = next(p for p in configuration["Plugins"] if p["name"] == "HwTrace")
        config = plugin["values"]["archProfileUnits"]["gaudi2"]["HMMU"]["DCORE0_HMMU0_CS"]["bmonParams"]["b10"]
        capture_cycles = config["capture_window"]["value"] + 1
        bandwidth_cycles = config["bandwidth_window"]["value"] + 1
        factor = capture_cycles / bandwidth_cycles * 128
        result["response_counter"] = dict(beat_bytes=128, capture_cycles=capture_cycles,
                                          bandwidth_cycles=bandwidth_cycles, read_bytes=0, write_bytes=0,
                                          read_rounding_envelope=0, write_rounding_envelope=0, ports={})
        for name, events in beats.items():
            side = "read" if "Read" in name else "write"
            total = sum(e["counter_values"][2] for e in events) * factor
            envelope = sum(e["counter_values"][1] > 0 for e in events) * factor
            result["response_counter"][f"{side}_bytes"] += total
            result["response_counter"][f"{side}_rounding_envelope"] += envelope
            result["response_counter"]["ports"][name] = dict(bytes=total, samples=len(events),
                avg_exceeds_max=sum(e["counter_values"][2] > e["counter_values"][1] for e in events))
        for side in ("read", "write"):
            known = result.get(f"known_{side}_bytes", 0)
            result["response_counter"][f"{side}_known_ratio"] = result["response_counter"][f"{side}_bytes"] / known if known else None
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = {}
    for directory in sorted(args.root.iterdir()):
        if ((directory / "run.json").exists() or (directory / "window.json").exists()) and (directory / "native-events.jsonl").exists():
            rows[directory.name] = summarize(directory)
            record = rows[directory.name]
            print(directory.name, {k: record[k] for k in
                  ("median_seconds", "profiled_seconds", "integrated_bytes", "read_ratio", "write_ratio") if k in record},
                  {k: v for k, v in record.get("response_counter", {}).items() if k != "ports"})
    (args.root / "calibration-summary.json").write_text(json.dumps(rows, indent=2))
