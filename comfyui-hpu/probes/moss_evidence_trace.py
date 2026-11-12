"""Attribute device interval unions and retain fused-kernel identities."""
import argparse
import collections
import json
from pathlib import Path
import re


parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
args = parser.parse_args()


def classify(event):
    name = event['name'].lower()
    path = event.get('args', {}).get('EventName', '').lower()
    if 'gemm' in name:
        if 'fp8' in path:
            # Current block uses the unnumbered root for FFN down and 1..24 for AV.
            return 'fp8_linear' if '/proj/' in path or 'ffn_' in path or path.startswith('fp8_gemm_v2/') else 'fp8_av'
        return 'bf16_qk' if re.match(r'mm(?:_\d+)?/',path) or 'matmul' in path or 'bmm' in path else 'other_mme'
    if 'softmax' in name:
        return 'softmax'
    if 'dma' in name or 'memcpy' in name:
        return 'layout_dma'
    if 'fused_kernel' in name:
        return 'fused_tpc'
    if 'cast' in name or 'amax' in name or 'reduce_max' in name:
        return 'quantization_cast_reduce'
    if 'norm' in name:
        return 'norm'
    return 'other_named'


priority = ('fp8_linear', 'fp8_av', 'bf16_qk', 'other_mme', 'softmax')
result = []
for source in sorted(args.directory.glob('*.trace.json')):
    data = json.loads(source.read_text())
    events = data['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X' and e.get('dur', 0)>0]
    if not kernels:
        continue
    edges = []
    identities = {}
    for e in kernels:
        key = (classify(e), e['name'])
        identities[key] = dict(name=e['name'], category=key[0], path=e.get('args',{}).get('EventName'),
                               dtype=e.get('args',{}).get('dataType'), recipe_id=e.get('args',{}).get('recipeId'))
        edges.extend([(e['ts'],key,1),(e['ts']+e['dur'],key,-1)])
    edges.sort()
    active = collections.Counter()
    budget, others, unions = collections.Counter(), collections.Counter(), collections.Counter()
    previous = edges[0][0]
    for stamp, key, change in edges:
        elapsed = stamp-previous
        live = [key for key,count in active.items() if count]
        categories = {key[0] for key in live}
        selected = next((p for p in priority if p in categories), None)
        label = selected or ('other_kernel' if live else 'idle')
        budget[label] += elapsed
        for key_live in live:
            unions[key_live] += elapsed
        if label == 'other_kernel':
            others[' + '.join(sorted(name for _,name in live))] += elapsed
        active[key] += change
        if active[key] == 0:
            del active[key]
        previous=stamp
    span = edges[-1][0]-edges[0][0]
    assert abs(sum(budget.values())-span)<.01
    assert abs(sum(others.values())-budget['other_kernel'])<.01
    record=dict(source=str(source),span_ms=span/1000,
                budget_ms={k:v/1000 for k,v in budget.items()},
                other_exclusive_combinations_ms={k:v/1000 for k,v in others.most_common()},
                kernel_unions=[dict(identities[k],union_ms=v/1000) for k,v in unions.most_common()],
                runtime_counts=dict(collections.Counter(e['name'] for e in events if e.get('cat')=='privateuse1_runtime')),
                base_time_ns=data.get('baseTimeNanoseconds'))
    result.append(record)
    (args.directory/'evidence-traces.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k not in ('kernel_unions','other_exclusive_combinations_ms')}),flush=True)
