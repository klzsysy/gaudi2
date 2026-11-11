"""Serial full-pipeline, fixed-caption, quality, release, and natural-text checks."""
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(os.environ['MOSS_COMPLETION_ROOT'])
probes=Path(__file__).parent
source=(root.parent/'performance-2026-11-11-bf16/baseline/source.json').read_text()
base=dict(os.environ,PT_HPU_LAZY_MODE='0',GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',PYTHONPATH=str(Path.cwd()))
for name,extra in [('baseline',dict(MOSS_NORM_EAGER='1')),('candidate',dict(MOSS_NORM_AB='1',MOSS_BF16_ACCEPTANCE='1',MOSS_RELEASE='1'))]:
    if name not in os.environ.get('MOSS_PIPELINE_CASES', 'baseline,candidate').split(','):
        continue
    directory=root/name;directory.mkdir(exist_ok=True)
    (directory/'source.json').write_text(source)
    env=dict(base,**extra,MOSS_PIPELINE_DIR=str(directory))
    with (root/(name+'-pipeline.log')).open('w') as log:
        subprocess.run([sys.executable,str(probes/'moss_pipeline_measure.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(name,'pipeline PASS',flush=True)
env=dict(base,MOSS_DIT_ROOT=str(root/'final-fixed'),MOSS_DIT_CASES='product',MOSS_DIT_CACHE_TEXT='1',MOSS_DIT_RUNS='6',
         MOSS_CAPTION=str(root.parent/'performance-2026-11-10-dit-next/caption.txt'))
with (root/'final-fixed.log').open('w') as log:
    subprocess.run([sys.executable,str(probes/'run_moss_dit_trials.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print('fixed caption PASS',flush=True)
for name,extra in [('baseline',dict(MOSS_NORM_EAGER='1')),('candidate',{})]:
    env=dict(base,**extra,MOSS_RESULT_DIR=str(root/('natural-'+name)),MOSS_WORKFLOW=str(root/'baseline/source.json'),MOSS_GDN_MODE='product',MOSS_SEED='1',MOSS_LENGTH='4096',MOSS_REPEATS='1')
    with (root/('natural-'+name+'.log')).open('w') as log:
        subprocess.run([sys.executable,str(probes/'moss_norm_natural.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(name,'natural PASS',flush=True)
subprocess.run([sys.executable,str(probes/'summarize_moss_bf16_round.py'),str(root)],check=True)
