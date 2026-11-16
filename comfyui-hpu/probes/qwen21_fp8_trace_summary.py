"""Extract FP8 work and device activity from record-shapes HPU traces."""
import argparse
import collections
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('trace', type=Path)
a = parser.parse_args()
events = json.loads(a.trace.read_text())['traceEvents']


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X' and e.get('dur', 0) > 0]
groups = collections.defaultdict(list)
for e in kernels:
    name = e['name']
    kind = 'fp8_gemm' if 'gemm' in name.lower() and ('fp8' in name.lower() or 'f8' in name.lower()) else 'other_gemm' if 'gemm' in name.lower() else 'softmax' if 'softmax' in name.lower() else 'other_kernel'
    groups[kind].append((e['ts'], e['ts'] + e['dur']))
active = union([(e['ts'], e['ts'] + e['dur']) for e in kernels])
shapes = collections.Counter()
flops = 0
for e in events:
    if e.get('name') != 'hpu::fp8_gemm_v2' or e.get('ph') != 'X':
        continue
    dims = [d for d in e.get('args', {}).get('Input Dims', []) if isinstance(d, list) and len(d) >= 2]
    if len(dims) >= 2:
        x, w = dims[:2]
        shapes[(tuple(x), tuple(w))] += 1
        flops += 2 * math.prod(x[:-1]) * x[-1] * w[0]
span = active[-1][1] - active[0][0]
report = dict(source=str(a.trace), fp8_linear_flops=flops,
              fp8_shapes=[dict(input=k[0], weight=k[1], count=v) for k,v in shapes.items()],
              kernel_names=collections.Counter(e['name'] for e in kernels),
              device_span_ms=span/1000,
              device_active_ms=sum(b-a for a,b in active)/1000,
              group_union_ms={k:sum(b-a for a,b in union(v))/1000 for k,v in groups.items()},
              runtime_counts=collections.Counter(e['name'] for e in events if e.get('cat')=='privateuse1_runtime'),
              fp8_cpu_events=sum(e.get('name')=='hpu::fp8_gemm_v2' and e.get('ph')=='X' for e in events))
covered = []
exclusive = {}
for kind in ('fp8_gemm', 'other_gemm', 'softmax', 'other_kernel'):
    combined = union(covered + union(groups[kind]))
    exclusive[kind] = (sum(b-a for a,b in combined) - sum(b-a for a,b in covered)) / 1000
    covered = combined
report['exclusive_ms'] = exclusive
report['device_gap_ms'] = report['device_span_ms']-report['device_active_ms']
a.trace.with_suffix('.fp8-summary.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('kernel_names','fp8_shapes')}, indent=2))
print('kernel names', report['kernel_names'].most_common(25))
