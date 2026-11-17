"""Join node wall-clock intervals with sampled device and host telemetry."""
import argparse
import collections
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
args = parser.parse_args()
root = args.root
telemetry = [json.loads(line) for line in (root / 'telemetry.jsonl').open()]
previous = {}
for sample in telemetry:
    for process in sample.get('processes', []):
        prior = previous.get(process['pid'])
        if prior:
            timestamp, old = prior
            seconds = (sample['wall_ns'] - timestamp) / 1e9
            process['cpu_cores'] = (process['cpu_seconds'] - old['cpu_seconds']) / seconds
            old_threads = {thread['id']: thread['cpu_seconds'] for thread in old['threads']}
            process['busiest_thread_cores'] = max(
                [(thread['cpu_seconds'] - old_threads.get(thread['id'], thread['cpu_seconds'])) / seconds
                 for thread in process['threads']], default=0)
        previous[process['pid']] = (sample['wall_ns'], process)


def describe(values):
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values), samples=len(values)) if values else {}


def hardware(start, end):
    samples = [s for s in telemetry if start <= s['wall_ns'] <= end and 'hpu' in s]
    result = {}
    for key in ('power.draw', 'utilization.aip', 'memory.used', 'temperature.aip', 'clocks.current.soc'):
        values = []
        for sample in samples:
            try:
                values.append(float(sample['hpu'][key]))
            except ValueError:
                pass
        result[key] = describe(values)
    for key in ('cpu_cores', 'busiest_thread_cores', 'rss'):
        # Older captures also included the small orchestration process.
        servers = [max(sample['processes'], key=lambda process: process['rss'])
                   for sample in samples if sample.get('processes')]
        values = [process[key] for process in servers if key in process]
        result[key] = describe(values)
    result['violation_counters'] = {key: sorted({s['hpu'][key] for s in samples})
                                    for key in ('stats.violation.power', 'stats.violation.thermal')}
    return result


requests = collections.defaultdict(lambda: dict(stages={}, intervals={}, executed=[]))
starts = {}
for line in (root / 'workflow/events.jsonl').open():
    event = json.loads(line)
    pid = event['prompt_id']
    row = requests[pid]
    kind = event['kind']
    if kind == 'request_start':
        row['start_ns'] = event['wall_ns']
    elif kind == 'request_end':
        row['end_ns'] = event['wall_ns']
        row['seconds'] = (row['end_ns'] - row['start_ns']) / 1e9
        row['fallback_counts'] = event['fallback_counts']
        row['memory_stats'] = event['memory_stats']
    elif kind == 'node_start':
        starts[(pid, event['node'])] = event['wall_ns']
        row['executed'].append(event['class_type'])
    elif kind == 'node_end':
        start = starts[(pid, event['node'])]
        name = event['class_type']
        row['stages'][name] = row['stages'].get(name, 0) + (event['wall_ns'] - start) / 1e9
        row['intervals'].setdefault(name, []).append([start, event['wall_ns']])
    elif kind == 'generate_start':
        row['input_tokens'] = event['token_shapes']
        row['generate_start_ns'] = event['wall_ns']
    elif kind == 'progress' and event.get('class_type') == 'TextGenerate':
        row.setdefault('token_times_ns', {})[event['value']] = event['wall_ns']
    elif kind == 'generate_end':
        row['output_tokens'] = event['output_tokens']
        row['token_ids'] = event['token_ids']
        row['first_token_seconds'] = (row['token_times_ns'][1] - row['generate_start_ns']) / 1e9
        tokenizer = Path('/home/sonny/ComfyUI-Gaudi/comfy/text_encoders/qwen35_tokenizer/tokenizer_config.json')
        config = json.loads(tokenizer.read_text())
        close_token = next(int(key) for key, value in config['added_tokens_decoder'].items()
                           if value['content'] == '</think>')
        if close_token in row['token_ids']:
            boundary = row['token_ids'].index(close_token) + 1
            timestamp = row['token_times_ns'][boundary]
            row['thinking'] = dict(tokens_including_close=boundary,
                                   following_tokens=len(row['token_ids']) - boundary,
                                   seconds=(timestamp - row['generate_start_ns']) / 1e9,
                                   following_seconds=(event['wall_ns'] - timestamp) / 1e9)

runs = {}
for path in sorted((root / 'workflow').glob('*.json')):
    history = json.loads(path.read_text())
    if not isinstance(history, dict) or 'prompt' not in history:
        continue
    assert history['status']['status_str'] == 'success'
    row = requests[history['prompt'][1]]
    row['hardware'] = {name: hardware(intervals[0][0], intervals[-1][1])
                       for name, intervals in row['intervals'].items()}
    if path.name.startswith('full-'):
        assert {'TextGenerate', 'TextEncodeQwenImage21', 'SamplerCustomAdvanced', 'VAEDecode'}.issubset(row['executed'])
    runs[path.stem] = row

warm = [row for label, row in runs.items() if label.startswith('full-') and label != 'full-0']
report = dict(cold_seconds=runs['full-0']['seconds'],
              warm_seconds=describe([row['seconds'] for row in warm]),
              stage_medians={name: statistics.median(row['stages'][name] for row in warm)
                             for name in warm[0]['stages']},
              cached_seconds=describe([row['seconds'] for name, row in runs.items()
                                       if name.startswith('cached-') and 'TextGenerate' not in row['executed']]),
              runs=runs)
(root / 'latest-summary.json').write_text(json.dumps(report, indent=2))
print(json.dumps({key: value for key, value in report.items() if key != 'runs'}, indent=2))
