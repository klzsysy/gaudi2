"""Bracket an independent-head SDPA workflow trial with current-product runs."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ['MOSS_NEXT_ROOT'])
for name,settings in [('head-baseline-a',{}),('head-candidate',dict(MOSS_SDPA_HEAD_ONE='1')),('head-baseline-b',{})]:
    if name not in os.environ.get('MOSS_HEAD_PIPELINE_CASES','head-baseline-a,head-candidate,head-baseline-b').split(','):
        continue
    env = dict(os.environ,**settings,MOSS_DIT_ROOT=str(root/name),MOSS_DIT_CASES='product',MOSS_DIT_CACHE_TEXT='1',
               MOSS_CAPTION=str(root.parent/'performance-2026-11-10-dit-next/caption.txt'))
    with (root/(name+'.log')).open('w') as log:
        subprocess.run([sys.executable,str(Path(__file__).with_name('run_moss_dit_trials.py'))],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(name,json.loads((root/name/'product/summary.json').read_text()),flush=True)
