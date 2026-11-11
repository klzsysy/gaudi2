"""Summarize measured requests without mixing cold, diagnostic and cached runs."""
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for name in ('native-first', 'fixed-reference', 'fixed-native'):
    directory = root / name
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    requests = {}
    for end in (event for event in events if event['kind'] == 'request_end'):
        prompt_id = end['prompt_id']
        request = [event for event in events if event.get('prompt_id') == prompt_id]
        start = next(event for event in request if event['kind'] == 'request_start')
        starts = {event['node']: event for event in request if event['kind'] == 'node_start'}
        stages = {}
        for event in request:
            if event['kind'] == 'node_end':
                kind = event['class_type']
                stages[kind] = stages.get(kind, 0) + (event['mono_ns'] - starts[event['node']]['mono_ns']) / 1e9
        requests[prompt_id] = {
            'seconds': (end['mono_ns'] - start['mono_ns']) / 1e9,
            'stages': stages,
            'fallback': end['fallback_counts'],
            'memory': end['memory_stats'],
            'checks': [event for event in request if event['kind'] == 'tensor_check'],
            'generated_tokens': [event['output_tokens'] for event in request if event['kind'] == 'generate_end'],
        }
    runs = []
    for history in sorted(directory.glob('run-*-history.json')):
        record = json.loads(history.read_text())
        assert record['status']['status_str'] == 'success'
        prompt_id = record['prompt'][1]
        runs.append(requests[prompt_id])
    assert runs
    for request in requests.values():
        assert not request['fallback'], request['fallback']
        assert all(check['finite'] for check in request['checks'])
    summary[name] = {'cold_diagnostic': runs[0]['seconds'], 'warm_seconds': [run['seconds'] for run in runs[1:]],
                     'warm_median': statistics.median(run['seconds'] for run in runs[1:]),
                     'dit_median': statistics.median(run['stages']['KSampler'] for run in runs[1:]), 'requests': requests}
(root / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps({name: {key: value for key, value in result.items() if key != 'requests'} for name, result in summary.items()}, indent=2))
