"""Summarize fixed-workflow A/B, lifecycle, and PE hardware evidence."""
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
api = root / 'api'
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
        record = stages.setdefault(event['prompt_id'], {})
        record[event['class_type']] = record.get(event['class_type'], 0.) + elapsed
    elif event['kind'] == 'request_end':
        ends[event['prompt_id']] = event
for row in rows:
    row['stage_seconds'] = stages[row['prompt_id']]
    row['memory_stats'] = ends[row['prompt_id']]['memory_stats']
    row['fallback_counts'] = ends[row['prompt_id']]['fallback_counts']
fixed = {}
for baseline, name in ((True, 'baseline'), (False, 'product')):
    selected = [row for row in rows if row['label'].startswith('fixed-') and row['baseline'] == baseline]
    fixed[name] = dict(runs=len(selected), median_seconds=statistics.median(row['seconds'] for row in selected),
                       range_seconds=[min(row['seconds'] for row in selected), max(row['seconds'] for row in selected)],
                       stages={name: statistics.median(row['stage_seconds'][name] for row in selected)
                               for name in ('TextGenerate', 'TextEncodeQwenImage21', 'SamplerCustomAdvanced',
                                            'VAEDecode', 'SaveImageAdvanced')})
fixed['full_reduction'] = 1 - fixed['product']['median_seconds'] / fixed['baseline']['median_seconds']
fixed['pe_reduction'] = 1 - fixed['product']['stages']['TextGenerate'] / fixed['baseline']['stages']['TextGenerate']
trace = json.loads((root / 'trace/text.trace.fp8-summary.json').read_text())
trace_times = [json.loads(line) for line in (root / 'trace/timings.jsonl').read_text().splitlines()]
plain = statistics.median(row['window_seconds'] for row in trace_times if row['mode'] == 'plain')
profiled = next(row['window_seconds'] for row in trace_times if row['mode'] == 'profile')
trace['plain_window_seconds'] = plain
trace['profiled_window_seconds'] = profiled
trace['profiler_overhead'] = profiled / plain - 1
trace['kernel_coverage'] = trace['device_active_ms'] / trace['device_span_ms']
trace['tokens'] = 64
trace['recipes_per_token'] = trace['runtime_counts']['launch_recipe'] / 64
trace['launches_per_token'] = trace['runtime_counts']['launch'] / 64
trace['main_thread_ms_per_token'] = statistics.median(row['window_thread_seconds'] for row in trace_times
                                                     if row['mode'] == 'plain') * 1000 / 64
telemetry = []
invalid_telemetry = 0
for line in (root / 'telemetry.jsonl').read_text().splitlines():
    try:
        telemetry.append(json.loads(line))
    except json.JSONDecodeError:
        invalid_telemetry += 1
hardware_samples = [row['hpu'] for row in telemetry if 'hpu' in row]
telemetry_summary = {field: [min(float(row[field]) for row in hardware_samples),
                             max(float(row[field]) for row in hardware_samples)]
                     for field in ('power.draw', 'temperature.aip', 'clocks.current.soc',
                                   'stats.violation.power', 'stats.violation.thermal')}
summary = dict(source=str(root), fixed=fixed, requests=rows, trace=trace,
               hardware=json.loads((root / 'hardware-summary.json').read_text()),
               logits=json.loads((root / 'logits-comparison.json').read_text()),
               reload=json.loads((api / 'reload-equivalence.json').read_text()),
               unload=json.loads((api / 'unload.json').read_text()),
               telemetry_partial=telemetry_summary if invalid_telemetry == 0 else None,
               invalid_telemetry_records=invalid_telemetry,
               validated_sha256=json.loads((root / 'validated-hashes.json').read_text()))
(root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(dict(fixed=fixed, trace={key: trace[key] for key in ('profiler_overhead', 'kernel_coverage',
                 'recipes_per_token', 'launches_per_token', 'main_thread_ms_per_token')},
                 natural=[{key: row[key] for key in ('label', 'seconds', 'tokens')} for row in rows
                          if row['label'] in ('natural-baseline', 'natural-product', 'seed-baseline', 'seed-product')],
                 telemetry=telemetry_summary), indent=2))
