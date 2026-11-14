"""Collect compact timing, numeric and pixel evidence for the November 15 run."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
a = p.parse_args()
root = a.root
report = dict(workflows=json.loads((root/'summary.json').read_text()), text={}, blocks={}, traces={}, pixels=[])
for directory in sorted(root.glob('text-*')):
    if not directory.is_dir() or not (directory/'timings.jsonl').exists():
        continue
    rows = [json.loads(line) for line in (directory/'timings.jsonl').read_text().splitlines()]
    plain = [row['seconds'] for row in rows if row['mode']=='plain']
    item = dict(timings=rows, median=statistics.median(plain) if plain else None)
    quality = root/(directory.name+'-quality.json')
    if quality.exists():
        item['logits'] = json.loads(quality.read_text())
    report['text'][directory.name] = item
for path in root.glob('block-*/result.json'):
    report['blocks'][path.parent.name] = json.loads(path.read_text())
for path in root.glob('*/*.trace.summary.json'):
    report['traces'][path.parent.name] = json.loads(path.read_text())

def pixels(history):
    data = json.loads(history.read_text())
    output = data['outputs']['461']['images'][0]
    path = Path('/home/sonny/ComfyUI-Gaudi/output')/output['subfolder']/output['filename']
    return np.asarray(Image.open(path).convert('RGBA'))

expected = pixels(root/'latest-baseline/full-1.json')
for label in ('latest-baseline', 'latest-product', 'latest-reference-repeat', 'latest-final'):
    for path in sorted((root/label).glob('full-*.json')):
        actual = pixels(path)
        report['pixels'].append(dict(case=f'{label}/{path.name}', shape=list(actual.shape),
                                     exact=np.array_equal(expected, actual),
                                     sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
report['tests'] = {path.name:path.read_text().splitlines()[-1] for path in root.glob('product-tests*.log')}
report['product_files'] = {str(path.relative_to('/home/sonny/ComfyUI-Gaudi')):hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (Path('/home/sonny/ComfyUI-Gaudi/comfy/text_encoders')/name for name in ('qwen35.py','hpu_mlp.py'))}
report['weight_read_model'] = dict(decoder_bytes=13839123456, lm_head_bytes=2034237440,
                                 historical_copy_tb_s=2.148, ideal_weight_read_ms=(13839123456+2034237440)/2.148e9,
                                 scope='Ideal sequential weight read only, excludes KV/GDN, TPC, layout and scheduling')
for name in ('cpu-comparison', 'natural-quality', 'normal-quality', 'text-product-quality', 'mtp-fresh-comparison', 'mtp-regression', 'mtp-baseline-regression'):
    path = root/(name+'.json')
    if path.exists():
        report[name] = json.loads(path.read_text())
if (root/'normal').exists():
    report['normal'] = {}
    for name in ('cold', 'seed2-512', 'reload', 'final'):
        path = root/'normal'/(name+'.json')
        if path.exists():
            history = json.loads(path.read_text())
            stamps = [entry[1]['timestamp'] for entry in history['status']['messages'] if entry[0] in ('execution_start','execution_success')]
            report['normal'][name] = dict(status=history['status']['status_str'],seconds=(stamps[-1]-stamps[0])/1000)
    for name in ('queue', 'free'):
        path = root/'normal'/(name+'.json')
        if path.exists():
            report['normal'][name] = json.loads(path.read_text())
(root/'release-evidence.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(pixels=report['pixels'],tests=report['tests']),indent=2))
