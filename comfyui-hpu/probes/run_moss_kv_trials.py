"""Run KV candidates serially with identical generation settings."""
import json
import os
from pathlib import Path
import subprocess
import sys


root = Path(os.environ['MOSS_EXPERIMENT_DIR'])
for mode in os.environ.get('MOSS_KV_MODES', 'baseline,compile-full,compile-prefix').split(','):
    destination = root / (os.environ.get('MOSS_TRIAL_PREFIX', 'text64') + '-' + mode)
    destination.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, MOSS_KV_MODE=mode, MOSS_RESULT_DIR=str(destination), MOSS_GDN_MODE='product',
               MOSS_WORKFLOW=str(root / 'probes/moss-user-textgen-bf16.json'),
               MOSS_LENGTH=os.environ.get('MOSS_LENGTH', '64'), MOSS_REPEATS=os.environ.get('MOSS_REPEATS', '5'),
               MOSS_KV_CAPACITY=os.environ.get('MOSS_KV_CAPACITY', '6073'))
    with (destination / 'process.log').open('w') as stream:
        subprocess.run([sys.executable, str(root / 'probes/moss_kv_experiment.py')], env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(mode, (destination / 'summary.json').read_text(), flush=True)
