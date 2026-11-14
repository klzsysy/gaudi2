"""Finish fixed-logit, natural-image and ordinary-service acceptance serially."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--wait-pid', type=int)
p.add_argument('--pipeline-screen', action='store_true')
a = p.parse_args()
if a.wait_pid:
    while Path(f'/proc/{a.wait_pid}').exists():
        time.sleep(5)

base = Path('/home/sonny/gaudi2/comfyui-hpu')
root = base/'performance-qwen21-2026-11-15'
assert (root/'latest-final/free.json').exists(), 'Final workflow acceptance did not finish'
previous = base/'performance-qwen21-bf16-2026-11-13'
env = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so', PYTHONPATH=str(Path.cwd()))
if a.pipeline_screen:
    for flag in ('1', '0'):
        label = 'text-pipeline-'+flag
        with (root/(label+'.log')).open('w') as log:
            subprocess.run([sys.executable, str(base/'probes/qwen21_text_profile.py'), str(root/label),
                            '--teacher', str(previous/'teacher.json'), '--logits'],
                           env=dict(env, PT_HPU_EAGER_PIPELINE_ENABLE=flag), stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        print(label, (root/label/'timings.jsonl').read_text(), flush=True)
    raise SystemExit(0)
product_text = root/'text-pipeline-1'
assert (product_text/'logits.pt').exists(), 'Run the paired pipeline screen first'
with (root/'text-product-cpu.log').open('w') as log:
    subprocess.run([sys.executable, str(base/'probes/qwen21_text_profile.py'), str(root/'text-product-cpu'),
                    '--workflow', str(base/'workflows/qwen21-latest-2026-11-15-api.json'),
                    '--window-tokens', '64', '--cpu-profile'],
                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
subprocess.run([sys.executable, str(base/'probes/qwen21_deep_jobs.py'), 'final-quality', '--output-root', str(root),
                '--runs', '0', '--cached', '0', '--quality-cases'], env=env, check=True)

import torch
import numpy as np
from PIL import Image

expected = torch.load(root/'text-reference/logits.pt', weights_only=True).float()
for name in ('text-product', 'text-rope-cache'):
    directory = product_text if name == 'text-product' else root/name
    actual = torch.load(directory/'logits.pt', weights_only=True).float()
    result = dict(exact=torch.equal(expected, actual), max_abs=(actual-expected).abs().max().item(),
                  rrmse=((actual-expected).square().mean()/expected.square().mean()).sqrt().item(),
                  top1_equal=(actual.argmax(-1)==expected.argmax(-1)).sum().item())
    (root/(name+'-quality.json')).write_text(json.dumps(result, indent=2))
    if name == 'text-product':
        (root/'text-pipeline-1-quality.json').write_text(json.dumps(result, indent=2))
    if name == 'text-product':
        assert result['exact'], result

def pixels(history):
    item = history['outputs']['461']['images'][0]
    return np.asarray(Image.open(Path('/home/sonny/ComfyUI-Gaudi/output')/item['subfolder']/item['filename']).convert('RGBA'))

def tokens(directory):
    result = {}
    for line in (directory/'events.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event['kind']=='generate_end':
            result[event['prompt_id']] = event['token_ids']
    return result

old_tokens, new_tokens = tokens(previous/'final-product'), tokens(root/'final-quality')
quality = []
for label in ('seed2', 'size512', 'size1024', 'thinking', 'official'):
    old = json.loads((previous/'final-product'/f'{label}-0.json').read_text())
    new = json.loads((root/'final-quality'/f'{label}-0.json').read_text())
    same_tokens = old_tokens[old['prompt'][1]] == new_tokens[new['prompt'][1]]
    same_pixels = np.array_equal(pixels(old), pixels(new))
    quality.append(dict(case=label, tokens_exact=same_tokens, pixels_exact=same_pixels,
                        output_tokens=len(new_tokens[new['prompt'][1]])))
(root/'natural-quality.json').write_text(json.dumps(quality, indent=2))
assert all(row['tokens_exact'] and row['pixels_exact'] for row in quality), quality
print(quality, flush=True)
subprocess.run(['./start-hpu.sh'], check=True)
subprocess.run([sys.executable, str(base/'probes/qwen21_normal_next.py')], env=env, check=True)
expected_image = pixels(json.loads((root/'latest-baseline/full-1.json').read_text()))
normal_quality = {label:np.array_equal(expected_image, pixels(json.loads((root/'normal'/f'{label}.json').read_text())))
                  for label in ('cold','reload')}
(root/'normal-quality.json').write_text(json.dumps(normal_quality, indent=2))
assert all(normal_quality.values()), normal_quality
