"""Aggregate request/node timing and verify matching generated-token work."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser()
p.add_argument('root',type=Path)
a=p.parse_args()
report={}
for directory in sorted(a.root.iterdir()):
    if not directory.is_dir() or not (directory/'events.jsonl').exists():continue
    groups=collections.defaultdict(list)
    for line in (directory/'events.jsonl').read_text().splitlines():
        event=json.loads(line);groups[event['prompt_id']].append(event)
    rows=[]
    for file in sorted(directory.glob('*.json')):
        if file.name.endswith('.trace.json') or file.name=='summary.json':continue
        data=json.loads(file.read_text())
        if not isinstance(data,dict) or 'status' not in data:continue
        messages=data['status']['messages']
        start=next(m[1] for m in messages if m[0]=='execution_start')
        finish=next(m[1] for m in messages if m[0] in ('execution_success','execution_error'))
        row=dict(file=file.name,prompt_id=start['prompt_id'],status=data['status']['status_str'],seconds=(finish['timestamp']-start['timestamp'])/1000,nodes={})
        starts={}
        for e in groups[row['prompt_id']]:
            if e['kind']=='node_start':starts[e['node']]=e['mono_ns']
            if e['kind']=='node_end':row['nodes'][e['class_type']]=(e['mono_ns']-starts[e['node']])/1e9
            if e['kind']=='generate_start':row['input_tokens']=e['token_shapes']
            if e['kind']=='generate_end':
                row['output_tokens']=e['output_tokens']
                row['token_sha256']=hashlib.sha256(json.dumps(e['token_ids']).encode()).hexdigest()
            if e['kind']=='request_end':row.update(fallback=e['fallback_counts'],peak_gib=e['memory_stats']['MaxInUse']/2**30,end_gib=e['memory_stats']['InUse']/2**30)
        rows.append(row)
    full=[r for r in rows if r['file'].startswith('full-') and r['file']!='full-0.json' and r['status']=='success']
    cached=[r for r in rows if r['file'].startswith('cached-') and 'TextGenerate' not in r['nodes'] and r['status']=='success']
    summary={'runs':rows}
    for label,group in [('warm_full',full),('cached_text',cached)]:
        if group:
            summary[label]={'count':len(group),'median':statistics.median(r['seconds'] for r in group),'range':[min(r['seconds'] for r in group),max(r['seconds'] for r in group)],
                            'nodes':{kind:statistics.median(r['nodes'][kind] for r in group if kind in r['nodes']) for kind in set().union(*(r['nodes'] for r in group))}}
    if (directory/'diagnostics.jsonl').exists():summary['diagnostics']=[json.loads(x) for x in (directory/'diagnostics.jsonl').read_text().splitlines()]
    report[directory.name]=summary
    print(directory.name,{k:v for k,v in summary.items() if k in ['warm_full','cached_text']},flush=True)
(a.root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
