"""Summarize warm full workflows with phase telemetry at the server's CST timezone."""
import csv
import datetime
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
summary = {}
for directory in sorted(root.glob('full-*')):
    if not (directory / 'runs.jsonl').exists():
        continue
    runs = [json.loads(line) for line in (directory / 'runs.jsonl').read_text().splitlines()]
    warm = [r for r in runs if r['label'].startswith('b0-warm') and not r.get('diagnostic', False)]
    if not warm:
        continue
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    samples = []
    for row in list(csv.reader((directory / 'hpu.csv').open()))[1:]:
        if len(row) != 5:
            continue
        timestamp = datetime.datetime.strptime(row[0].replace(' CST', ''), '%a %b %d %H:%M:%S %Y').replace(tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
        samples.append((timestamp, *map(float, row[1:])))
    result = {'warm_count': len(warm), 'seconds': [r['request_seconds'] for r in warm],
              'median_seconds': statistics.median(r['request_seconds'] for r in warm),
              'tokens': [r['output_tokens'] for r in warm], 'phases': {}}
    for phase in ('request', 'TextGenerate', 'KSampler'):
        selected = []
        durations = []
        for run in warm:
            relevant = [e for e in events if e.get('prompt_id') == run['prompt_id'] and (phase == 'request' or e.get('class_type') == phase)]
            start = next(e for e in relevant if e['kind'] == ('request_start' if phase == 'request' else 'node_start'))
            end = next(e for e in relevant if e['kind'] == ('request_end' if phase == 'request' else 'node_end'))
            durations.append((end['mono_ns'] - start['mono_ns']) / 1e9)
            selected.extend(s for s in samples if start['wall_ns'] / 1e9 <= s[0] and s[0] + 1 <= end['wall_ns'] / 1e9)
        result['phases'][phase] = {'median_seconds': statistics.median(durations), 'aip_mean': statistics.mean(s[1] for s in selected),
                                   'aip_peak': max(s[1] for s in selected), 'power_54V_mean': statistics.mean(s[4] for s in selected), 'samples': len(selected)}
    summary[directory.name] = result
(root / 'phase-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
