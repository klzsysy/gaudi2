"""Use the same teacher tokens for each KV/decode implementation."""
import json
import os
from pathlib import Path
import subprocess
import sys


root = Path(os.environ['MOSS_EXPERIMENT_DIR'])
cases = [('control', 'original', '', '1'), ('kv', 'baseline', '', '1'), ('attention', 'baseline', 'attention', '1'), ('delta', 'baseline', 'delta', '1'), ('both', 'baseline', 'both', '1')]
if os.environ.get('MOSS_TEACHER_FINAL') == '1':
    cases = [('final-control', 'original', '', '1'), ('final-product', 'baseline', '', '0')]
for name, kv_mode, region, disable_attention in cases:
    destination = root / ('teacher-' + name)
    destination.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, MOSS_KV_MODE=kv_mode, MOSS_KV_OBSERVE='0', MOSS_RESULT_DIR=str(destination),
               MOSS_GDN_MODE='product', MOSS_DECODE_REGIONS=region,
               MOSS_DISABLE_ATTENTION_REGION=disable_attention,
               MOSS_WORKFLOW=str(root / 'probes/moss-user-textgen-bf16.json'), MOSS_MTP='off',
               MOSS_TEACHER=str(root / 'region-none/runs.jsonl'), MOSS_LENGTH='32', MOSS_REPEATS='5', MOSS_KV_CAPACITY='6073')
    with (destination / 'process.log').open('w') as stream:
        result = subprocess.run([sys.executable, str(root / 'probes/moss_kv_experiment.py')], env=env, stdout=stream, stderr=subprocess.STDOUT)
    summary = json.loads((destination / 'summary.json').read_text()) if (destination / 'summary.json').exists() else None
    print(json.dumps({'name': name, 'exit_code': result.returncode, 'summary': summary}), flush=True)
