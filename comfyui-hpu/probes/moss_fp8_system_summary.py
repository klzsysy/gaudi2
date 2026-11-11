"""Attribute sampled HPU intervals and repeated workflow node times."""
import collections
import json
from pathlib import Path
import statistics
import sys


def classify(event):
    name = event['name'].lower()
    path = event.get('args', {}).get('EventName', '').lower()
    if 'gemm' in name:
        return 'fp8_mme' if 'fp8' in path else 'other_mme'
    if 'softmax' in name:
        return 'softmax'
    if 'fp8' in name and 'cast' in name:
        return 'fp8_cast'
    return 'other_kernel'


def trace(source):
    events = json.loads(source.read_text())['traceEvents']
    kernels = [event for event in events if event.get('cat') == 'kernel' and event.get('ph') == 'X' and event.get('dur', 0) > 0]
    edges, names = [], collections.Counter()
    for event in kernels:
        category = classify(event)
        edges.extend([(event['ts'], category, 1), (event['ts'] + event['dur'], category, -1)])
        names[event['name']] += event['dur'] / 1000
    edges.sort()
    current, durations = collections.Counter(), collections.Counter()
    previous = edges[0][0]
    for stamp, category, change in edges:
        label = '+'.join(sorted(key for key, count in current.items() if count)) or 'idle'
        durations[label] += (stamp - previous) / 1000
        current[category] += change
        previous = stamp
    return {'span_ms': (edges[-1][0] - edges[0][0]) / 1000, 'disjoint_ms': dict(durations),
            'kernels': len(kernels), 'summed_kernel_ms': dict(names.most_common(20)),
            'runtime_counts': dict(collections.Counter(event['name'] for event in events if event.get('cat') == 'privateuse1_runtime'))}


root = Path(sys.argv[1])
result = {}
for target in root.iterdir():
    if not (target / 'runs.jsonl').exists() or not (target / 'events.jsonl').exists():
        continue
    runs = [json.loads(line) for line in (target / 'runs.jsonl').read_text().splitlines()]
    events = [json.loads(line) for line in (target / 'events.jsonl').read_text().splitlines()]
    stages = collections.defaultdict(list)
    warm = [run for run in runs if run['label'].startswith('full-warm')]
    for run in warm:
        request = [event for event in events if event.get('prompt_id') == run['prompt_id']]
        starts = {event['node']: event for event in request if event['kind'] == 'node_start'}
        times = collections.Counter()
        for event in request:
            if event['kind'] == 'node_end':
                times[event['class_type']] += (event['mono_ns'] - starts[event['node']]['mono_ns']) / 1e9
        for key, value in times.items():
            stages[key].append(value)
    result[target.name] = {'warm_seconds': [run['request_seconds'] for run in warm],
                           'median': statistics.median(run['request_seconds'] for run in warm) if warm else None,
                           'node_medians': {key: statistics.median(value) for key, value in stages.items()},
                           'traces': {} if '--timings-only' in sys.argv else {str(path.relative_to(target)): trace(path) for path in target.rglob('*.trace.json')}}
(root / ('timing-summary.json' if '--timings-only' in sys.argv else 'system-summary.json')).write_text(json.dumps(result, indent=2))
print(json.dumps({key: {k: v for k, v in value.items() if k != 'traces'} for key, value in result.items()}, indent=2))
