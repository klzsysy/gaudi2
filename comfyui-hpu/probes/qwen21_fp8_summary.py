"""Summarize complete recomputations separately from diagnostic captures."""
import argparse
import collections
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
a = parser.parse_args()
report = {}
for directory in sorted(a.root.iterdir()):
    if directory.name.startswith('native-') or not (directory / 'events.jsonl').exists():
        continue
    requests = collections.defaultdict(lambda: dict(stages=collections.defaultdict(float), executed=[]))
    starts = {}
    for line in (directory / 'events.jsonl').open():
        event = json.loads(line)
        pid = event['prompt_id']
        row = requests[pid]
        kind = event['kind']
        if kind == 'request_start':
            starts[pid] = event['mono_ns']
        if kind == 'request_end':
            row['seconds'] = (event['mono_ns'] - starts[pid]) / 1e9
            row['fallback'] = event['fallback_counts']
            row['memory'] = event['memory_stats']
        if kind == 'node_start':
            starts[(pid, event['node'])] = event['mono_ns']
            row['executed'].append(event['class_type'])
        if kind == 'node_end':
            row['stages'][event['class_type']] += (event['mono_ns'] - starts[(pid, event['node'])]) / 1e9
        if kind == 'generate_start':
            row['input_tokens'] = event['token_shapes']
        if kind == 'generate_end':
            row['output_tokens'] = event['output_tokens']
            row['token_ids'] = event['token_ids']
            row['text'] = event['text']
    runs = []
    for path in sorted(directory.glob('full-*.json')):
        history = json.loads(path.read_text())
        assert history['status']['status_str'] == 'success'
        row = dict(requests[history['prompt'][1]])
        row['history'] = str(path)
        row['images'] = [image for output in history['outputs'].values() for image in output.get('images', [])]
        present = {node['class_type'] for node in history['prompt'][2].values()}
        required = present & {'TextGenerate', 'TextEncodeQwenImage21', 'SamplerCustomAdvanced', 'VAEDecode'}
        assert required.issubset(row['executed'])
        runs.append(row)
    if not runs:
        continue
    warm = runs[1:]
    medians = {name: statistics.median(row['stages'][name] for row in warm)
               for name in warm[0]['stages']} if warm else {}
    report[directory.name] = dict(cold_seconds=runs[0]['seconds'], warm_samples=len(warm),
        warm_median=statistics.median(row['seconds'] for row in warm) if warm else None,
        stage_medians=medians, runs=runs)
    print(directory.name, report[directory.name]['cold_seconds'], report[directory.name]['warm_median'],
          medians, [row.get('output_tokens') for row in runs])
(a.root / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
