"""Repeat the head-grouping A/B through the product's model-local dispatcher."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ['MOSS_NEXT_ROOT'])
for name,settings in [('head-product-baseline',dict(MOSS_HEAD_BASELINE='1')),('head-product-candidate',{})]:
    env = dict(os.environ,**settings,MOSS_DIT_ROOT=str(root/name),MOSS_DIT_CASES='product',MOSS_DIT_CACHE_TEXT='1',
               MOSS_CAPTION=str(root.parent/'performance-2026-11-10-dit-next/caption.txt'))
    with (root/(name+'.log')).open('w') as log:
        subprocess.run([sys.executable,str(Path(__file__).with_name('run_moss_dit_trials.py'))],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(name,json.loads((root/name/'product/summary.json').read_text()),flush=True)
