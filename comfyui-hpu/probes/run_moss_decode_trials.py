"""Compare larger decode regions serially after installing the KV candidate."""
import json
import os
from pathlib import Path
import subprocess
import sys


root = Path(os.environ['MOSS_EXPERIMENT_DIR'])
for region in os.environ.get('MOSS_REGION_CASES', 'none,attention,delta,both').split(','):
    destination = root / ('region-' + region)
    destination.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, MOSS_KV_MODE='baseline', MOSS_KV_OBSERVE='0', MOSS_RESULT_DIR=str(destination),
               MOSS_GDN_MODE='product', MOSS_DECODE_REGIONS='' if region == 'none' else region,
               MOSS_WORKFLOW=str(root / 'probes/moss-user-textgen-bf16.json'),
               MOSS_LENGTH='64', MOSS_REPEATS='3', MOSS_KV_CAPACITY='6073')
    with (destination / 'process.log').open('w') as stream:
        result = subprocess.run([sys.executable, str(root / 'probes/moss_kv_experiment.py')], env=env, stdout=stream, stderr=subprocess.STDOUT)
    summary = json.loads((destination / 'summary.json').read_text()) if (destination / 'summary.json').exists() else None
    print(json.dumps({'region': region, 'exit_code': result.returncode, 'summary': summary}), flush=True)
