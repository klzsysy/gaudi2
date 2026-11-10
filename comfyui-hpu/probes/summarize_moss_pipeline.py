"""Reconcile current pipeline nodes, cache cases and separate synchronized diagnostics."""
import collections
import csv
import datetime
import json
from pathlib import Path
import statistics
import sys


root = Path(sys.argv[1])
runs = [json.loads(line) for line in (root / 'runs.jsonl').read_text().splitlines()]
events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
source = json.loads((root / 'source.json').read_text())
telemetry = []
for row in list(csv.reader((root / 'hpu.csv').open()))[1:]:
    if len(row) != 5:
        continue
    timestamp = datetime.datetime.strptime(row[0].replace(' CST', ''), '%a %b %d %H:%M:%S %Y').replace(tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
    telemetry.append((timestamp, *map(float, row[1:])))


def intervals(relevant, kind, key):
    active = collections.defaultdict(list)
    result = []
    for event in relevant:
        name = event.get(key)
        if event['kind'] == kind + '_start':
            active[name].append(event)
        elif event['kind'] == kind + '_end':
            opening = active[name].pop()
            result.append({'name': name, 'node': event.get('node'), 'class_type': event.get('class_type'),
                           'start_ns': opening['mono_ns'], 'end_ns': event['mono_ns'],
                           'wall_start_ns': opening['wall_ns'], 'wall_end_ns': event['wall_ns'],
                           'seconds': (event['mono_ns'] - opening['mono_ns']) / 1e9})
    assert not any(active.values()), active
    return result


def nonoverlapping(rows):
    ordered = sorted(rows, key=lambda r: r['start_ns'])
    assert all(a['end_ns'] <= b['start_ns'] for a, b in zip(ordered, ordered[1:])), 'Overlapping regions cannot be added'


rows = []
for run in runs:
    relevant = [e for e in events if e.get('prompt_id') == run['prompt_id']]
    begin = next(e for e in relevant if e['kind'] == 'request_start')
    end = next(e for e in relevant if e['kind'] == 'request_end')
    nodes = intervals(relevant, 'node', 'node')
    nonoverlapping(nodes)
    by_node = collections.defaultdict(float)
    for node in nodes:
        assert begin['mono_ns'] <= node['start_ns'] <= node['end_ns'] <= end['mono_ns']
        by_node[node['name']] += node['seconds']
    wall_spans = {'request': [(begin['wall_ns'], end['wall_ns'])]}
    for node in nodes:
        wall_spans.setdefault(node['class_type'], []).append((node['wall_start_ns'], node['wall_end_ns']))
    row = dict(run, nodes=[{'node': nid, 'class_type': source[nid]['class_type'], 'seconds': seconds,
                           'percent': seconds / run['request_seconds'] * 100} for nid, seconds in by_node.items()],
               outside_nodes_seconds=run['request_seconds'] - sum(by_node.values()), wall_spans=wall_spans)
    if run['diagnostic']:
        regions = intervals(relevant, 'region', 'region')
        summary = {}
        for node_id in by_node:
            matching = [r for r in regions if r['node'] == node_id]
            if not matching:
                continue
            grouped = collections.defaultdict(list)
            for region in matching:
                grouped[region['name']].append(region['seconds'])
            summary[node_id] = {name: {'calls': len(values), 'seconds': sum(values)} for name, values in grouped.items()}
            if source[node_id]['class_type'] == 'TextGenerate':
                nonoverlapping(matching)
                summary[node_id]['unclassified'] = {'seconds': by_node[node_id] - sum(r['seconds'] for r in matching)}
            if source[node_id]['class_type'] == 'KSampler':
                children = [r for r in matching if r['name'] != 'dit_model_forward']
                nonoverlapping(children)
                model_seconds = summary[node_id]['dit_model_forward']['seconds']
                summary[node_id]['other_model_work'] = {'seconds': model_seconds - sum(r['seconds'] for r in children)}
                summary[node_id]['outside_model'] = {'seconds': by_node[node_id] - model_seconds}
        row['diagnostic_regions'] = summary
    rows.append(row)

result = {'metadata': json.loads((root / 'metadata.json').read_text()), 'groups': {},
          'cold': next(r for r in rows if r['label'] == 'full-cold'),
          'transition': next(r for r in rows if r['label'] == 'cached-transition'),
          'diagnostic': next(r for r in rows if r['diagnostic'])}
for group in ('full', 'cached'):
    chosen = [r for r in rows if r['label'].startswith(group + '-warm')]
    selected = sorted(chosen, key=lambda r: r['request_seconds'])[len(chosen) // 2]
    metrics = {}
    for phase in ('request', 'TextGenerate', 'KSampler'):
        samples = [sample for run in chosen for start, end in run['wall_spans'].get(phase, [])
                   for sample in telemetry if start / 1e9 <= sample[0] and sample[0] + 1 <= end / 1e9]
        if samples:
            metrics[phase] = {'aip_mean': statistics.mean(s[1] for s in samples), 'aip_peak': max(s[1] for s in samples),
                              'power_54V_mean': statistics.mean(s[4] for s in samples), 'samples': len(samples)}
    result['groups'][group] = {'count': len(chosen), 'seconds': [r['request_seconds'] for r in chosen],
                               'tokens': [r['output_tokens'] for r in chosen],
                               'median_seconds': statistics.median(r['request_seconds'] for r in chosen),
                               'selected': selected, 'telemetry': metrics,
                               'node_medians': {nid: statistics.median(next((n['seconds'] for n in r['nodes'] if n['node'] == nid), 0) for r in chosen)
                                                for nid in source}}
(root / 'summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
