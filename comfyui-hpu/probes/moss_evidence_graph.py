"""Map compiled fused kernels and attention storage back to graph operations."""
import argparse
import collections
import json
from pathlib import Path
import re


parser=argparse.ArgumentParser()
parser.add_argument('root',type=Path)
args=parser.parse_args()
mapping={}
storage=[]
sources=list((args.root/'block-native/post').rglob('*.post.json'))
for folder in args.root.glob('block-native-*/post'):
    sources.extend(folder.rglob('*.post.json'))
for source in sources:
    graph=json.loads(source.read_text())['graphs'][0]
    tensors={t['name']:t for t in graph['tensors']}
    for node in graph['nodes']:
        if 'fused_node_graph' in node:
            operations=node['fused_node_graph']['nodes']
            guids=[n['guid'] for n in operations]
            paths=[n['name'] for n in operations]
            combined=' '.join(guids)
            if 'tanh' in combined and 'sigmoid' in combined:
                category='SiTU'
            elif 'rope' in combined:
                category='RoPE_and_cast'
            elif 'reduce_sum_f32' in combined:
                category='QK_FP32_norm'
            elif 'reduce_sum_bf16' in combined:
                category='modulation_norm'
            elif 'abs_fwd' in combined or 'reduce_max' in combined or 'to_hf8' in combined:
                category='quantization'
            elif 'cast_i32_to_bf16' in combined:
                category='modulation_coefficients'
            elif any('add_fwd_f32'==g for g in guids):
                category='residual'
            elif guids==['mult_f32','cast_f32_to_bf16','mult_bf16']:
                category='GEMM_output_scale'
            else:
                category='unresolved_fused'
            match=re.search(r'0x[0-9A-F]+',node['guid'])
            if match:
                mapping[match[0]]=dict(category=category,guid=node['guid'],graph=graph['name'],operations=guids,paths=paths)
        if node['engine']=='MME' or 'softmax' in node['guid']:
            storage.append(dict(node=node['name'],engine=node['engine'],guid=node['guid'],
                inputs=[{k:tensors[n].get(k) for k in ['name','allocation','dtype','max_shape','alias_of']} for n in node['input_tensors']],
                outputs=[{k:tensors[n].get(k) for k in ['name','allocation','dtype','max_shape','alias_of']} for n in node['output_tensors']]))
traces=json.loads((args.root/'dit-gc/evidence-traces.json').read_text())
attributed=collections.Counter()
unmapped=collections.Counter()
for trace in traces:
    for combination,ms in trace['other_exclusive_combinations_ms'].items():
        categories=[]
        for name in combination.split(' + '):
            match=re.search(r'0x[0-9A-F]+',name)
            if match:
                category=mapping.get(match[0],{}).get('category','unresolved_fused')
                if category=='unresolved_fused':unmapped[name]+=ms
            elif 'Dma' in name or 'memcpy' in name:
                category='layout_copy'
            else:
                category='named_'+name
            categories.append(category)
        attributed[' + '.join(sorted(set(categories)))]+=ms*25/3/1000
result=dict(kernel_mapping=mapping,attention_storage=storage,
            other_exclusive_seconds=dict(attributed.most_common()),unmapped_fused_ms=dict(unmapped))
(args.root/'graph-attribution.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result['other_exclusive_seconds'],indent=2))
print('unmapped',dict(unmapped))
