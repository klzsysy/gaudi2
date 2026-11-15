"""Summarize native execution-event coverage without treating it as occupancy."""
import argparse
import collections
import json
from pathlib import Path


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def duration(intervals):
    return sum(end - start for start, end in intervals)


def intersection(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start < end:
            result.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


parser = argparse.ArgumentParser()
parser.add_argument("events", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--tokens", type=int, default=16)
args = parser.parse_args()
stacks = collections.defaultdict(list)
engines = {}
events = collections.defaultdict(list)
unmatched = collections.Counter()
recipes = []
for line in args.events.open():
    event = json.loads(line)
    pid, phase, timestamp = event["pid"], event["ph"], event["ts"]
    if phase == "M" and event["name"] == "process_name":
        name = event.get("metadata_name", "").split(" ")[0].lstrip("*")
        if name in ("MME", "TPC", "EDMA", "PDMA"):
            engines[pid] = name
    if event["category"] == "Recipe Start":
        recipes.append((timestamp, timestamp + event["dur"]))
    if pid not in engines or phase not in ("B", "E"):
        continue
    key = (pid, event["tid"], event["id"])
    if phase == "B":
        stacks[key].append(timestamp)
    elif stacks[key]:
        start = stacks[key].pop()
        assert timestamp >= start
        events[engines[pid]].append((start, timestamp))
    else:
        unmatched[engines[pid]] += 1
compute = union(events["MME"] + events["TPC"] + events["EDMA"])
recipe_union = union(recipes)
span = compute[-1][1] - compute[0][0]
covered = duration(compute)
inside = duration(recipe_union) - duration(intersection(compute, recipe_union))
report = dict(
    source=str(args.events), tokens=args.tokens,
    engines={name: dict(engine_events=len(values), covered_ms=duration(union(values)) / 1000)
             for name, values in events.items()},
    compute_span_ms=span / 1000, compute_covered_ms=covered / 1000,
    compute_uncovered_ms=(span - covered) / 1000,
    recipe_count=len(recipes), recipes_per_token=len(recipes) / args.tokens,
    recipe_covered_ms=duration(recipe_union) / 1000,
    recipe_without_compute_events_ms=inside / 1000,
    unmatched_ends=dict(unmatched), unmatched_starts=sum(map(len, stacks.values())),
    interpretation="Instrumented execution-event intervals. Uncovered time includes scheduling, dependencies, and any activity absent from these event classes. It is not a calibrated idle counter.",
)
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
