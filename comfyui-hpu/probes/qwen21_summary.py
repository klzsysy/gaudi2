"""Summarize first-release Qwen workflow measurements and source provenance."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
results = root / 'results/qwen21-first-2026-11-13'
events = {}
for path in results.glob('*-events.jsonl'):
    for line in path.read_text().splitlines():
        item = json.loads(line)
        events.setdefault(item['prompt_id'], []).append(item)
report = {'date':'2026-11-13', 'base_product':'34e2649e', 'product':'9f2ff96e', 'tests':{'passed':80,'skipped':1}, 'runs':{}}
for folder in sorted(results.iterdir()):
    if not folder.is_dir():
        continue
    runs = []
    for path in sorted(folder.glob('[0-9]*.json'), key=lambda p:int(p.stem)):
        item = json.loads(path.read_text())
        if 'status' not in item:
            continue
        messages = item['status']['messages']
        begin = next(m[1] for m in messages if m[0]=='execution_start')
        end = next(m[1] for m in messages if m[0] in ('execution_success','execution_error'))
        row = dict(run=int(path.stem), prompt_id=begin['prompt_id'], status=item['status']['status_str'],
                   seconds=(end['timestamp']-begin['timestamp'])/1000,
                   images=item['outputs'].get('461',{}).get('images',[]))
        starts = {}
        row['nodes'] = {}
        for event in events.get(row['prompt_id'],[]):
            kind = event['kind']
            if kind == 'node_start':
                starts[event['node']] = event['mono_ns']
            elif kind == 'node_end':
                row['nodes'][event['class_type']] = (event['mono_ns']-starts[event['node']])/1e9
            elif kind == 'generate_end':
                row['generated_tokens'] = event['output_tokens']
                row['generated_text'] = event['text']
            elif kind == 'request_end':
                row['fallback_counts'] = event['fallback_counts']
                row['peak_gib'] = event['memory_stats']['MaxInUse']/2**30
                row['end_gib'] = event['memory_stats']['InUse']/2**30
        runs.append(row)
    if runs:
        report['runs'][folder.name] = runs
        print(folder.name, [(r['status'], r['seconds'], r.get('generated_tokens')) for r in runs])
report['notes'] = ['compiled and product folders are rejected attention-plus-FFN prototypes that reached Dynamo recompilation limits', 'enhanced-fixed run 0 overlaps CPU BF16 conversion and is excluded from timing comparisons', 'INT8 and BF16 natural text outputs have different token counts', 'quality runs include diagnostic replay and CPU FP64 work', 'normal-service warm requests change noise seed and reuse text node cache']
report['source_sha256'] = {}
product = root.parents[1] / 'ComfyUI-Gaudi'
for file in ['comfy/ldm/qwen_image21/model.py','comfy/text_encoders/llama.py','comfy/text_encoders/qwen35.py']:
    report['source_sha256'][file] = hashlib.sha256((product/file).read_bytes()).hexdigest()
report['workflow_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'workflows').glob('qwen21-*')}
for name in ('quality.jsonl', 'pe-conversion.json', 'free-stable.json', 'image-comparison.json'):
    path=results/name
    if path.exists():
        report[name]=[json.loads(line) for line in path.read_text().splitlines()] if path.suffix=='.jsonl' else json.loads(path.read_text())
(root / 'evidence/qwen21-first-2026-11-13.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
