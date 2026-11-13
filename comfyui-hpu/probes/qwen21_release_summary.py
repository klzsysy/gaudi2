"""Collect the BF16 round's timings, rejected trials, quality, and provenance."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('root',type=Path)
p.add_argument('destination',type=Path)
p.add_argument('--product',type=Path)
a=p.parse_args()
report={'date':'2026-11-13','baseline_product':'9f2ff96e','product':'ed9552e5','tests':{'passed':159,'skipped':1},'workflow':'qwen21-user-bf16-api.json',
        'settings':{'width':832,'height':1504,'steps':8,'sampler':'euler','scheduler':'simple','cfg':1,'thinking':False,'seed':1,'pe_seed':1},
        'measurements':json.loads((a.root/'summary.json').read_text())}
for name in ['model-manifest.json','release-quality.json','rounding-comparisons.json','image-comparisons.json','text-delta1-quality.json']:
    if (a.root/name).exists():report[name]=json.loads((a.root/name).read_text())
report['block_trials']={}
for directory in a.root.glob('block-*'):
    if (directory/'result.json').exists():report['block_trials'][directory.name]=json.loads((directory/'result.json').read_text())
report['text_windows']={}
for label in ['text-main-profile','text-teacher-baseline','text-teacher-delta1']:
    if (a.root/label/'timings.jsonl').exists():
        report['text_windows'][label]=[json.loads(line) for line in (a.root/label/'timings.jsonl').read_text().splitlines()]
quality=a.root/'product-quality/block-quality.jsonl'
if quality.exists():report['live_blocks']=[json.loads(line) for line in quality.read_text().splitlines()]
report['notes']=[
    'Warm full requests invalidate PE, encoding, sampling and decoding nodes while retaining loader outputs.',
    'First cached request can recompute text after fingerprint transition and is excluded from cached-text medians.',
    'HTTP-thread profiler attempt was terminated after a history timeout. Main-thread trace succeeded.',
    'The 16-token profile window had about 26% instrumentation overhead. Device-event gaps are not a proven removable budget.',
    'Per-module synchronization inflated the detailed decode diagnostic and is excluded from performance comparisons.',
    'GDN expansion changed fixed logits and natural text, and its no-decomposition variant failed lowering.',
    'Compiled masked-prefix attention changed image details. The accepted path compiles unmasked attention.',
    'Sampling compilation is restricted to temperature 1.0 after three BF16 non-unit-temperature exactness regressions.'
]
if a.product:
    report['product_sha256']={name:hashlib.sha256((a.product/name).read_bytes()).hexdigest() for name in [
        'comfy/ldm/qwen_image21/model.py','comfy/text_encoders/llama.py','tests-unit/comfy_test/hpu_qwen21_block_test.py','tests-unit/comfy_test/hpu_textgen_test.py']}
a.destination.parent.mkdir(parents=True,exist_ok=True)


normal=a.root/'normal'
if normal.exists():
    report['normal_service']={}
    for label in ['cold','new-seed','reload']:
        path=normal/(label+'.json')
        if path.exists():
            data=json.loads(path.read_text())
            stamps=[m[1]['timestamp'] for m in data['status']['messages'] if m[0] in ('execution_start','execution_success')]
            report['normal_service'][label]={'status':data['status']['status_str'],'seconds':(stamps[-1]-stamps[0])/1000}
    for label in ['free','queue']:
        if (normal/(label+'.json')).exists():report['normal_service'][label]=json.loads((normal/(label+'.json')).read_text())
a.destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
