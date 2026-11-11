"""Disjoint device-time attribution; never sum overlapping hardware lanes."""
import collections
import json
from pathlib import Path
import sys


def group(event):
    name = event['name']
    path = event.get('args', {}).get('EventName', '')
    if name in ('GEMM', 'gemm', 'BatchGemm', 'batch_gemm') and path.startswith('bmm_1/'):
        return 'av_mme'
    if name in ('GEMM', 'gemm', 'BatchGemm', 'batch_gemm') and path.startswith('bmm/'):
        return 'qk_mme'
    if name in ('GEMM', 'gemm'):
        return 'linear_mme'
    if name in ('BatchGemm', 'batch_gemm'):
        return 'attention_mme'
    if 'softmax' in name.lower():
        return 'softmax'
    return 'other_kernel'


def summarize(source):
    events = json.loads(source.read_text())['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X' and e.get('dur', 0) > 0]
    recipes = collections.defaultdict(list)
    for e in kernels:
        recipes[str(e.get('args', {}).get('recipeId'))].append(e.get('args', {}).get('EventName', ''))
    regions = {key: 'attention_region' if any('proj/qkv/' in name for name in names)
               else 'ffn_region' if any('ffn_up_gate/' in name for name in names)
               else 'unresolved_outer_ops' for key, names in recipes.items()}
    region_edges = []
    for e in kernels:
        category = regions[str(e.get('args', {}).get('recipeId'))]
        region_edges.extend(((e['ts'], category, 1), (e['ts'] + e['dur'], category, -1)))
    region_active, region_time = collections.Counter(), collections.Counter()
    region_edges.sort()
    previous = region_edges[0][0]
    for stamp, category, delta in region_edges:
        label = '+'.join(sorted(k for k, count in region_active.items() if count > 0)) or 'idle'
        region_time[label] += stamp - previous
        region_active[category] += delta
        previous = stamp
    edges = []
    for e in kernels:
        category = group(e)
        edges.extend(((e['ts'], category, 1), (e['ts'] + e['dur'], category, -1)))
    edges.sort()
    active = collections.Counter()
    exclusive = collections.Counter()
    previous = edges[0][0]
    for stamp, category, delta in edges:
        label = '+'.join(sorted(k for k, count in active.items() if count > 0)) or 'idle'
        exclusive[label] += stamp - previous
        active[category] += delta
        previous = stamp
    total = edges[-1][0] - edges[0][0]
    assert abs(sum(exclusive.values()) - total) < 0.01
    runtime = collections.Counter(e['name'] for e in events if e.get('cat') == 'privateuse1_runtime')
    matrix_union = sum(t for name, t in exclusive.items() if 'mme' in name)
    return {'source': str(source), 'device_span_ms': total / 1000,
            'device_active_union_ms': (total-exclusive['idle']) / 1000,
            'matrix_union_ms': matrix_union / 1000,
            'disjoint_ms': {k: v/1000 for k, v in sorted(exclusive.items())},
            'region_disjoint_ms': {k: v/1000 for k, v in sorted(region_time.items())},
            'runtime_counts': dict(runtime),
            'warning': 'Short profiler trace; extrapolation is a budget, not unprofiled phase timing.'}


root = Path(sys.argv[1])
result = [summarize(p) for p in sorted((root / 'captures').glob('forward-*.trace.json'))]
(root / 'trace-budget.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
