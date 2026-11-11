"""Compact evidence for the attention and deferred-GDN optimization round."""
import collections
import hashlib
import json
from pathlib import Path
import statistics
import sys
import numpy as np
from PIL import Image

root=Path(sys.argv[1])
checkout=Path.cwd()


def read(name):
    return json.loads((root/name).read_text())


def lines(name):
    return [json.loads(s) for s in (root/name).read_text().splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


result=dict(date='2026-11-11',product_commit=sys.argv[2],baseline_commit='64155fe4b9522b935702f674170ed27a8d25bb71',
            remote_evidence_root=str(root),prior_model_manifest='moss-bf16-2026-11-11.json',
            workflow_sha256=digest(root/'source.json'),
            program_sha256=digest(root.parent/'performance-2026-11-11-completion/mtp-record-298/program.pt'))
result['product_commits']=['41f01e68','700bdb6d','a8f42df2']
paths=['comfy/text_encoders/hpu_gated_delta.py','comfy/text_encoders/qwen35.py','comfy/ldm/moss_image/model.py',
       'tests-unit/comfy_test/hpu_gated_delta_test.py','tests-unit/comfy_test/hpu_moss_attention_test.py','tests-unit/comfy_test/hpu_gdn_region_test.py']
result['product_files']={name:digest(checkout/name) for name in paths}
result['head_screens']={name:read(name+'.json') for name in ['head-groups','head-layouts','attention-math']}
result['text_screens']={name:read(name+'/summary.json') for name in ['linear-baseline','linear-delta','linear-head','linear-all',
                       'pending-bucket','pending-exact','conv-control','tree-delta','tree-conv','product-text']}
result['numerical']=read('product-tree-numerical.json')
result['mtp_logits']=read('tree-logits-comparison.json')
result['region_logits']=read('region-logits-comparison.json')
result['compiler_rescreen']=read('delta-compile-compare.json')
result['region_text']={name:read(name+'/summary.json') for name in ['parallel-region','parallel-repeat','region-product-text']}
result['four_row_final']=read('four-product-parameters.json')
result['attention_backend_audit']={name:read(name+'.json') for name in ['attention-test-audit','attention-test-audit-cpu-selection']}
result['fixed_caption']={name:read(name+'/product/summary.json') for name in ['head-baseline-a','head-candidate','head-baseline-b',
                         'head-product-baseline','head-product-candidate']}
def pixels(case,index):
    path=root/case/'product'/f'image-{index}.png'
    return np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)


baseline=pixels('head-product-baseline',1)
candidate=pixels('head-product-candidate',1)
pixel_metrics={}
for name,left,right in [('candidate',candidate,baseline),('baseline_repeat',pixels('head-product-baseline',2),baseline),
                        ('candidate_repeat',pixels('head-product-candidate',2),candidate)]:
    delta=left-right
    pixel_metrics[name]=dict(mae=float(np.abs(delta).mean()),rmse=float(np.sqrt((delta*delta).mean())),maximum=float(np.abs(delta).max()))
(root/'head-product-pixels.json').write_text(json.dumps(pixel_metrics,indent=2))
result['fixed_caption_pixels']=pixel_metrics
matched=read('matched-pipeline-summary.json')
for value in matched.values():
    value['token_sha256']=hashlib.sha256(json.dumps(value.pop('ids')).encode()).hexdigest()
result['matched_pipeline']=matched
result['matched_final']=read('matched-final/summary.json')
result['submission_budget']=read('submission-budget.json')
natural=lines('natural-product/runs.jsonl')
warm=[r for r in natural if r['label'].startswith('full-warm')]
result['natural_pipeline']=dict(stage='Parallel recurrence and head splitting, before the wider GDN region',
                                warm_seconds=[r['request_seconds'] for r in warm],
                                median=statistics.median(r['request_seconds'] for r in warm),
                                tokens=[r['output_tokens'] for r in warm],cold=natural[0]['request_seconds'])
result['quality']=[dict(label=row['label'],tokens=row['tokens'],checks=row['checks']) for row in read('natural-product/quality/results.json')]
result['quality_image_differences']=read('natural-product/quality/image-differences.json')
result['release']={name:read(name+'/release/unload-series.json')[-1] for name in ['matched-baseline','matched-candidate','matched-final','natural-product']}
result['peak_bytes']={}
for name in ['matched-baseline','matched-candidate','matched-final','natural-product']:
    ends=[e for e in lines(name+'/events.jsonl') if e['kind']=='request_end'][:6]
    result['peak_bytes'][name]=max(e['memory_stats']['MaxInUse'] for e in ends)
result['tests']=(root/'tests.log').read_text().splitlines()[-1]
if (root/'restore-results.json').exists():
    result['restoration']=read('restore-results.json')
if (root/'deployment-check.json').exists():
    result['deployment']=read('deployment-check.json')
result['limits']=['Natural MTP captions can change with runtime decisions and BF16 rounding.',
                  'Head splitting is enabled for compiled, unmasked HPU BF16 batch-one MOSS attention with 24 heads of dimension 128 and sequence length >=4096.',
                  'The wider GDN region requires eight-row BF16 decode, ordinary operations, LinearKV, and no active hooks or gradients.',
                  'Four-row inputs are cloned outside the compiled recurrence after a full-model synStatus26 failure. The vendor-internal cause is unconfirmed.',
                  'Microbenchmarks and short profiler traces do not establish the hardware compute ceiling.']
(root/'next-evidence.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
print(root/'next-evidence.json')
