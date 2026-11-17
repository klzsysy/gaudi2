"""Summarize the startup regression, numerical comparison and staged deployment."""
import json
from pathlib import Path
import statistics
import sys


root = Path(sys.argv[1])
validation = root / 'validation'
api = validation / 'api'
rows = json.loads((api / 'results.json').read_text())
events = [json.loads(line) for line in (api / 'events.jsonl').read_text().splitlines()]
starts = {}
stages = {}
ends = {}
for event in events:
    key = event.get('prompt_id'), event.get('node')
    if event['kind'] == 'node_start':
        starts[key] = event['mono_ns']
    elif event['kind'] == 'node_end' and key in starts:
        elapsed = (event['mono_ns'] - starts[key]) / 1e9
        stage = stages.setdefault(event['prompt_id'], {})
        stage[event['class_type']] = stage.get(event['class_type'], 0.) + elapsed
    elif event['kind'] == 'request_end':
        ends[event['prompt_id']] = event
segments = (api / 'server.log').read_text(errors='replace').split('INFO:root:got prompt')[1:]
assert len(segments) == len(rows)
for row, segment in zip(rows, segments):
    row['stage_seconds'] = stages[row['prompt_id']]
    row['memory_stats'] = ends[row['prompt_id']]['memory_stats']
    row['fallback_counts'] = ends[row['prompt_id']]['fallback_counts']
    row['alias_warnings'] = segment.count('UserWarning: hpu::batch_as_strided')
    generation = json.loads((api / (row['prompt_id'] + '-generation.json')).read_text())
    row['synchronized_first_samples'] = generation['synchronized_first_samples']
    if row['synchronized_first_samples']:
        row['first_decode_iteration_seconds'] = row['sample_times'][1] - row['sample_times'][0]
    assert row['tokens_equal'] and row['caption_equal'] and row['rgba_equal'], row['label']
warm = {}
for baseline, name in ((True, 'baseline'), (False, 'product')):
    selected = [row for row in rows if row['label'].startswith('warm-') and row['baseline'] == baseline]
    assert len(selected) == 5
    assert all(not row['synchronized_first_samples'] for row in selected)
    warm[name] = dict(runs=len(selected), median_seconds=statistics.median(row['seconds'] for row in selected),
        range_seconds=[min(row['seconds'] for row in selected), max(row['seconds'] for row in selected)],
        stages={name: statistics.median(row['stage_seconds'][name] for row in selected)
                for name in ('TextGenerate', 'TextEncodeQwenImage21', 'SamplerCustomAdvanced', 'VAEDecode', 'SaveImageAdvanced')})
warm['full_change_fraction'] = warm['product']['median_seconds'] / warm['baseline']['median_seconds'] - 1
screen = json.loads((root / 'screen2/dynamic_compact/timings.json').read_text())
for row in screen:
    row['first_decode_iteration_seconds'] = row['first_samples'][1] - row['first_samples'][0]
    assert len(row['compilations']) == 1
summary = dict(source=str(root), tests='193 passed, 1 skipped, 28 warnings in 125.52s',
    warm=warm, requests=rows, capacity_screen=screen,
    logits=json.loads((root / 'logits-validation/logits-comparison.json').read_text()),
    unload=json.loads((api / 'unload.json').read_text()),
    validated_sha256=json.loads((validation / 'validated-hashes.json').read_text()),
    logits_validated_sha256=json.loads((root / 'logits-validation/validated-hashes.json').read_text()),
    deployment=json.loads((root / 'deployment/deployment.json').read_text()),
    limitations=['Fresh process or model reload still requires initial decoder compilation.',
                'This change preserves KV allocation capacity and graph math; hardware utilization was not reprofiled.',
                'The preliminary dynamic_compact run used an old getter in its copied namespace and is excluded.'])
assert summary['validated_sha256'] == summary['logits_validated_sha256']
(root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(dict(warm=warm, edits=[{key: row[key] for key in
    ('label', 'seconds', 'first_decode_iteration_seconds', 'alias_warnings')} for row in rows if row['label'].startswith('edit')],
    logits=summary['logits'], deployment=summary['deployment']['identity']), indent=2))
