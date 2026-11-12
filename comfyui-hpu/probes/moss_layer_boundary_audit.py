"""Read-only audit of framework overhead, transient attention storage and text gaps."""
import collections
import hashlib
import json
from pathlib import Path
import statistics
import sys


root = Path(sys.argv[1])
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
quality = root / 'performance-2026-11-12-quality-preserving'
evidence = root / 'performance-2026-11-12-evidence'


def merge_intervals(intervals):
    intervals.sort()
    start, end = intervals[0]
    first = start
    active = 0.0
    largest_gap = 0.0
    for left, right in intervals[1:]:
        if left > end:
            active += end - start
            largest_gap = max(largest_gap, left - end)
            start, end = left, right
        else:
            end = max(end, right)
    active += end - start
    return dict(first=first, last=end, span=end - first, active=active,
                gap=end - first - active, largest_gap=largest_gap)


pipeline = quality / 'pipeline-baseline'
runs = [json.loads(line) for line in (pipeline / 'runs.jsonl').read_text().splitlines()]
events = [json.loads(line) for line in (pipeline / 'events.jsonl').read_text().splitlines()]
rows = []
for run in runs:
    if not run['label'].startswith('full-warm'):
        continue
    selected = [event for event in events if event.get('prompt_id') == run['prompt_id']]
    starts, intervals, nodes = {}, [], collections.defaultdict(float)
    for event in selected:
        if event['kind'] == 'node_start':
            starts[event['node']] = event['mono_ns']
        elif event['kind'] == 'node_end':
            begin = starts.pop(event['node'])
            end = event['mono_ns']
            intervals.append((begin, end))
            nodes[event['class_type']] += (end - begin) / 1e9
    union = merge_intervals(intervals)['active'] / 1e9
    rows.append(dict(label=run['label'], request_seconds=run['request_seconds'], node_seconds=dict(nodes),
                     node_union_seconds=union, outside_nodes_seconds=run['request_seconds'] - union))
result = dict(workflow_rows=rows,
              median_outside_nodes_seconds=statistics.median(row['outside_nodes_seconds'] for row in rows),
              median_outside_nodes_fraction=statistics.median(row['outside_nodes_seconds'] / row['request_seconds'] for row in rows))

storage = json.loads((quality / 'attention-storage.json').read_text())
result['score_tensor_metadata'] = {}
result['fx_graphs'] = {}
for label in ['baseline_positive', 'baseline_negative', 'sram32']:
    graph_rows = []
    for item in storage[label]:
        source = Path(item['source'])
        graph = json.loads(source.read_text())['graphs'][0]
        tensors = [tensor for tensor in graph['tensors'] if tensor.get('dtype') == 'bf16'
                   and len(tensor.get('max_shape', [])) == 2
                   and tensor['max_shape'][0] == tensor['max_shape'][1] >= 4096]
        graph_rows.append(dict(source=str(source), graph=graph['name'], workspace_size=graph['workspace_size'],
                               scores=[{key: tensor.get(key) for key in ['name', 'max_shape', 'dtype', 'persistent',
                                                                       'in_persist_storage', 'allocation', 'allow_permutation', 'offset']}
                                       for tensor in tensors],
                               softmax_count=sum('softmax' in node.get('guid', '') for node in graph['nodes']),
                               mme_count=sum(node.get('engine') == 'MME' for node in graph['nodes'])))
    result['score_tensor_metadata'][label] = graph_rows
for label in ['block-native-full-tef', 'block-native-negative-tef']:
    source = evidence / label / 'fx-graphs.json'
    if source.exists():
        graphs = json.loads(source.read_text())
        result['fx_graphs'][label] = [dict(index=index, nodes=len(graph['nodes']),
                                         relevant_targets=dict(collections.Counter(node['target'] for node in graph['nodes']
                                                                                    if any(word in node['target'] for word in ['matmul', 'softmax', 'gemm']))))
                                    for index, graph in enumerate(graphs)]

package = Path('/opt/gaudi-install/venv-1.24.2-481-ubuntu2204/lib/python3.10/site-packages/habana_frameworks/torch')
files = ['version.py', 'dynamo/compile_backend/compilers.py', 'dynamo/compile_backend/_passes/fuse_attention.py',
         'dynamo/compile_backend/partitioner.py', 'dynamo/compile_backend/recipe_compiler.py']
result['installed_bridge'] = dict(version_source=(package / 'version.py').read_text(),
                                sha256={name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in files})
(output / 'audit.json').write_text(json.dumps(result, indent=2))
print('Workflow and storage audit saved', flush=True)

source = evidence / 'text-gc/10-profile16.trace.json'
trace = json.loads(source.read_text())
events = trace['traceEvents']
intervals = [(event['ts'], event['ts'] + event['dur']) for event in events
             if event.get('cat') == 'kernel' and event.get('ph') == 'X' and event.get('dur', 0) > 0]
runtime = dict(collections.Counter(event['name'] for event in events if event.get('cat') == 'privateuse1_runtime'))
summary = merge_intervals(intervals)
result['text_profile16'] = dict(source=str(source), kernel_events=len(intervals), runtime_counts=runtime,
                               **{key + '_ms': value / 1000 for key, value in summary.items() if key not in ['first', 'last']})
records = [json.loads(line) for line in (evidence / 'text-gc/perturbation.jsonl').read_text().splitlines()]
result['text_profile16']['controls'] = [record for record in records if record['mode'] in ['sync16', 'profile16']]
(output / 'audit.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result['text_profile16']), flush=True)
