"""Compact, replayable evidence manifest without raw traces or tensors."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
checkout = Path.cwd()

def read(relative):
    return json.loads((root/relative).read_text())

def lines(relative):
    return [json.loads(line) for line in (root/relative).read_text().splitlines()]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

result = dict(date='2026-11-11', timezone='Asia/Shanghai', product_commit=sys.argv[2], remote_evidence_root=str(root),
              workload=dict(input_tokens=1970, fixed_output_tokens=298, positive_tokens=307, negative_tokens=23,
                            pixels=[1024,1024], steps=25, cfg=2.5, sampler='er_sde', scheduler='simple', latent_seed=418625753878250),
              prior_model_manifest='moss-bf16-2026-11-11.json', program_sha256=digest(root/'mtp-record-298/program.pt'))
paths = ['comfy/text_encoders/qwen35.py','comfy/text_encoders/hpu_gated_delta.py','comfy/text_encoders/llama.py',
         'comfy/ldm/moss_image/hpu.py','comfy/ldm/moss_image/model.py','comfy/ops.py','tests-unit/comfy_test/hpu_qwen_norm_test.py']
result['product_file_sha256'] = {name:digest(checkout/name) for name in paths}
result['trace_budget'] = read('trace-budget.json')
result['rotating_budget'] = read('rotating-budget.json')
result['lazy_graph'] = read('lazy-graph.json')
result['lazy_chain'] = read('lazy-chain.json')
result['lazy_compile_error'] = read('lazy-compile-compatibility.json')['error'].split('ImportError:')[-1].split('\n')[0]
result['fixed_mtp'] = {}
for case in ['baseline','delta','both','norm','region','norm-region','norm-product','outer-norm','native-norm']:
    name = 'mtp-'+case
    rows = lines(name+'/runs.jsonl')
    result['fixed_mtp'][case] = dict(**read(name+'/summary.json'),cold=rows[0]['seconds'],
                                   fallback=rows[-1]['fallback'],peak_bytes=max(x['memory']['MaxInUse'] for x in rows))
for case in ['block','mtp','probability']:
    result['fixed_mtp']['final-region-'+case] = read('final-region-'+case+'/summary.json')
result['mtp_logits'] = read('mtp-logits-comparison.json')
result['outer_norm_logits'] = read('outer-norm-comparison.json')
result['norm_references'] = {name:read(name+'/norm-reference.json') for name in ['norm-reference','norm-qk-reference','native-norm-reference','native-norm-qk-reference']}
result['prefill_screens'] = {name:dict(prefill=read('prefill-'+name+'/summary.json'),encoding=read('prefill-'+name+'/encoding-summary.json')) for name in ['product','mlp','attention','both']}
result['natural_pipeline'] = read('summary-candidate.json')
matched = read('matched-pipeline-summary.json')
for item in matched.values():
    item['token_sha256'] = hashlib.sha256(json.dumps(item.pop('ids')).encode()).hexdigest()
result['matched_pipeline'] = matched
result['fixed_caption'] = read('final-fixed/product/summary.json')
result['threads16'] = lines('matched-threads16/runs.jsonl')
result['quality'] = read('candidate/quality/pixel-comparison.json')
result['natural_prompts'] = {}
for name in ['baseline','candidate']:
    result['natural_prompts'][name] = [{key:row[key] for key in ['name','repeat','seconds','tokens','natural_end']} for row in read('natural-'+name+'/natural.json')]
result['release'] = read('candidate/release/unload-series.json')[-1]
result['telemetry'] = {}
for name in ['baseline','candidate']:
    with (root/name/'hpu.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    stats = {}
    for column in rows[0]:
        if any(key in column for key in ['soc','power.draw','temperature']):
            values = [float(row[column]) for row in rows]
            stats[column.strip()] = dict(minimum=min(values),maximum=max(values))
    ends = [e for e in lines(name+'/events.jsonl') if e['kind']=='request_end'][:6]
    stats['full_pipeline_peak_bytes'] = max(e['memory_stats']['MaxInUse'] for e in ends)
    result['telemetry'][name] = stats
if (root/'final-text-tests.log').exists():
    result['final_test_summary'] = (root/'final-text-tests.log').read_text().splitlines()[-1]
if (root/'restore-results.json').exists():
    result['restoration'] = read('restore-results.json')
result['limits'] = ['Cached image target 14 seconds remains unmet.', 'Fixed-MTP full pipeline target 20 seconds remains unmet.',
                    'Natural MTP trajectories change with timing and numerical differences.',
                    'Paired images change beyond baseline repeat noise.', 'Convergence is scoped to the installed runtime, existing kernels and tested workload.']
(root/'completion-evidence.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
print(root/'completion-evidence.json')
