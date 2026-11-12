"""Summarize compiled softmax input/output storage, following tensor aliases."""
import argparse
import collections
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
options = parser.parse_args()
records = []
for source in sorted(options.root.rglob('*.post.json')):
    for graph in json.loads(source.read_text()).get('graphs', []):
        tensors = {tensor['name']: tensor for tensor in graph['tensors']}
        softmax_nodes = [node for node in graph['nodes'] if 'softmax' in node.get('guid', '')]
        if not softmax_nodes:
            continue
        allocations = collections.Counter()
        shapes = collections.Counter()
        for node in softmax_nodes:
            for role, names in [('input', node['input_tensors'][:1]), ('output', node['output_tensors'][:1])]:
                for name in names:
                    tensor = tensors[name]
                    shapes[(role, tuple(tensor.get('max_shape', [])))] += 1
                    visited = set()
                    while tensor.get('alias_of') in tensors and tensor['name'] not in visited:
                        visited.add(tensor['name'])
                        tensor = tensors[tensor['alias_of']]
                    allocations[(role, str(tensor.get('allocation')))] += 1
        records.append(dict(source=str(source), graph=graph['name'], workspace=graph.get('workspace_size'),
                            softmax_nodes=len(softmax_nodes),
                            allocations=[dict(role=k[0], allocation=k[1], count=v) for k, v in allocations.items()],
                            shapes=[dict(role=k[0], shape=k[1], count=v) for k, v in shapes.items()]))
assert records, 'No compiled softmax graphs found'
(options.root.parent / 'attention-storage.json').write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2))
