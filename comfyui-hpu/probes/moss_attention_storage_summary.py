"""Summarize QK and softmax storage in post-compilation graphs."""
import collections
import json
from pathlib import Path
import sys


root = Path(sys.argv[1])
directories = dict(sram32=root / 'post-sram32',
                   baseline_positive=root.parent / 'performance-2026-11-12-evidence/block-native-full-tef/post',
                   baseline_negative=root.parent / 'performance-2026-11-12-evidence/block-native-negative-tef/post')
result = {}
for label, directory in directories.items():
    records = []
    for path in directory.rglob('*.post.json'):
        for graph in json.loads(path.read_text()).get('graphs', []):
            softmax = [node for node in graph['nodes'] if 'softmax' in node.get('guid', '')]
            if not softmax:
                continue
            tensors = {tensor['name']: tensor for tensor in graph['tensors']}

            def storage(name):
                tensor = tensors[name]
                base = tensors.get(tensor.get('alias_of'), {})
                return dict(dtype=tensor['dtype'], shape=tensor['max_shape'], allocation=tensor.get('allocation'),
                            alias_shape=base.get('max_shape'), alias_allocation=base.get('allocation'))

            qk = [node for node in graph['nodes'] if node.get('engine') == 'MME'
                  and len(node['input_tensors']) >= 2
                  and all(tensors[name]['dtype'] == 'bf16' for name in node['input_tensors'][:2])]
            counters = {}
            for key, names in dict(qk_outputs=[name for node in qk for name in node['output_tensors']],
                                   softmax_inputs=[node['input_tensors'][0] for node in softmax],
                                   softmax_outputs=[node['output_tensors'][0] for node in softmax]).items():
                count = collections.Counter(json.dumps(storage(name), sort_keys=True) for name in names)
                counters[key] = [dict(json.loads(value), count=number) for value, number in count.items()]
            probabilities = {node['output_tensors'][0] for node in softmax}
            consumers = sum(node.get('engine') == 'MME' and bool(probabilities.intersection(node['input_tensors']))
                            for node in graph['nodes'])
            records.append(dict(source=str(path), qk_nodes=len(qk), softmax_nodes=len(softmax),
                                probability_mme_consumers=consumers, **counters))
    result[label] = records
(root / 'attention-storage.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
