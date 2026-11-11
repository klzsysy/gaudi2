"""Compare matched full and cached requests; keep diagnostics out of warm timings."""
import collections
import hashlib
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
candidate = sys.argv[2] if len(sys.argv) > 2 else 'candidate'
summary = {}
trajectories = {}
for name in ('baseline', candidate):
    directory = root / name
    runs = [json.loads(line) for line in (directory / 'runs.jsonl').read_text().splitlines()]
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    by_request = collections.defaultdict(list)
    for event in events:
        by_request[event['prompt_id']].append(event)
    cases = {}
    for case in ('full-warm', 'cached-warm'):
        samples = [r for r in runs if r['label'].startswith(case)]
        assert len(samples) == 5, (name, case, len(samples))
        stages = collections.defaultdict(list)
        ids = []
        for run in samples:
            entries = by_request[run['prompt_id']]
            starts = {e['node']: e for e in entries if e['kind'] == 'node_start'}
            times = collections.Counter()
            for e in entries:
                if e['kind'] == 'node_end':
                    times[e['class_type']] += (e['mono_ns'] - starts[e['node']]['mono_ns']) / 1e9
            for label, seconds in times.items():
                stages[label].append(seconds)
            ids.extend(e['token_ids'] for e in entries if e['kind'] == 'generate_end')
            assert all(not e['fallback_counts'] for e in entries if e['kind'] == 'request_end')
        seconds = [r['request_seconds'] for r in samples]
        cases[case] = dict(samples=seconds, median=statistics.median(seconds), minimum=min(seconds), maximum=max(seconds),
                           stage_medians={k: statistics.median(v) for k, v in stages.items()},
                           tokens=[len(v) for v in ids], ids_equal=all(v == ids[0] for v in ids),
                           token_sha256=[hashlib.sha256(json.dumps(v).encode()).hexdigest() for v in ids])
        if case == 'full-warm':
            trajectories[name] = ids
    cases['cold_seconds'] = runs[0]['request_seconds']
    cases['diagnostic_checks'] = [e for e in events if e['kind'] == 'tensor_check']
    summary[name] = cases
summary['all_warm_tokens_match'] = all(a == b for a, b in zip(trajectories['baseline'], trajectories[candidate]))
summary['full_delta_percent'] = 100 * (summary[candidate]['full-warm']['median'] / summary['baseline']['full-warm']['median'] - 1)
(root / ('summary-' + candidate.replace('/', '-') + '.json')).write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
