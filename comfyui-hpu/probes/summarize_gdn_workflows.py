"""Summarize full workflow timings, token counts and stage telemetry."""
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
    runs = [json.loads(x) for x in (directory / 'runs.jsonl').read_text().splitlines()]
    events = [json.loads(x) for x in (directory / 'events.jsonl').read_text().splitlines()]
    samples = []
    for row in list(csv.reader((directory / 'hpu.csv').open()))[1:]:
        if len(row) != 5:
            continue
        timestamp = datetime.datetime.strptime(row[0].replace(' CST', ''), '%a %b %d %H:%M:%S %Y').replace(tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
        samples.append((timestamp, *[float(x) for x in row[1:]]))
    records = []
    for run in runs:
        selection = [x for x in events if x.get('prompt_id') == run['prompt_id']]
        generated = [x for x in selection if x['kind'] in ('generate_start', 'generate_end')]
        end = next(x for x in selection if x['kind'] == 'request_end')
        starts = {x['node']: x for x in selection if x['kind'] == 'node_start'}
        stages = {}
        for item in selection:
            if item['kind'] == 'node_end':
                stages.setdefault(item['class_type'], 0)
                stages[item['class_type']] += (item['mono_ns'] - starts[item['node']]['mono_ns']) / 1e9
        duration = (generated[1]['mono_ns'] - generated[0]['mono_ns']) / 1e9
        lo, hi = generated[0]['wall_ns'] / 1e9, generated[1]['wall_ns'] / 1e9
        text_samples = [x for x in samples if x[0] >= lo and x[0]+1 <= hi]
        record = {'label': run['label'], 'diagnostic': run['diagnostic'], 'seconds': run['request_seconds'], 'generation_seconds': duration,
                  'tokens': generated[1]['output_tokens'], 'ids': generated[1]['token_ids'],
                  'tokens_per_second_including_prefill': generated[1]['output_tokens']/duration,
                  'stages': stages, 'fallback': end['fallback_counts'],
                  'max_HPU_allocator_bytes': end['memory_stats']['MaxInUse']}
        if text_samples:
            record['telemetry'] = {'samples': len(text_samples), 'mean_aip': statistics.mean(x[1] for x in text_samples),
                                   'max_aip': max(x[1] for x in text_samples), 'mean_power_W': statistics.mean(x[4] for x in text_samples),
                                   'zero_fraction': sum(x[1] == 0 for x in text_samples) / len(text_samples)}
        records.append(record)
    warm = [x for x in records if 'warm' in x['label'] and not x['diagnostic']]
    summary[directory.name] = {'runs': records, 'warm_count': len(warm),
                              'warm_median': statistics.median(x['seconds'] for x in warm) if warm else None,
                              'generation_median': statistics.median(x['generation_seconds'] for x in warm) if warm else None,
                              'ids_equal': all(x['ids'] == records[0]['ids'] for x in records)}
    print(directory.name, json.dumps({k:v for k,v in summary[directory.name].items() if k != 'runs'}), flush=True)
(root / 'workflow-summary.json').write_text(json.dumps(summary, indent=2))
