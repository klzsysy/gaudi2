"""Summarize device interval unions and labelled host regions from Qwen traces."""
import argparse
import collections
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('traces', nargs='+', type=Path)
a = p.parse_args()

def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged

for path in a.traces:
    events = json.loads(path.read_text())['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X' and e.get('dur', 0) > 0]
    active = union([(e['ts'], e['ts']+e['dur']) for e in kernels])
    span = active[-1][1]-active[0][0]
    gaps = sorted((right[0]-left[1] for left, right in zip(active, active[1:])), reverse=True)
    mme = union([(e['ts'], e['ts']+e['dur']) for e in kernels if 'gemm' in e['name'].lower()])
    host = collections.defaultdict(list)
    for e in events:
        if e.get('ph') == 'X' and e.get('name', '').startswith('PE_'):
            host[e['name']].append((e['ts'], e['ts']+e.get('dur', 0)))
    report = dict(source=str(path), device_span_ms=span/1000,
                  device_active_ms=sum(end-start for start, end in active)/1000,
                  mme_union_ms=sum(end-start for start, end in mme)/1000,
                  longest_gaps_ms=[gap/1000 for gap in gaps[:20]],
                  host_regions_ms={name:sum(end-start for start,end in union(values))/1000 for name,values in host.items()},
                  runtime_counts=dict(collections.Counter(e['name'] for e in events if e.get('cat')=='privateuse1_runtime')))
    report['device_gap_ms'] = report['device_span_ms']-report['device_active_ms']
    path.with_suffix('.summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report), flush=True)
