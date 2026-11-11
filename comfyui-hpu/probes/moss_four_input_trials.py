"""Locate the runtime input property absent from CPU replay captures."""
import os
from pathlib import Path
import subprocess
import sys

root=Path(os.environ['MOSS_NEXT_ROOT'])
base=dict(os.environ,PT_HPU_LAZY_MODE='0',GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',PYTHONPATH=str(Path.cwd()),
          MOSS_WORKFLOW=str(root/'source.json'),MOSS_GDN_MODE='product',MOSS_LENGTH='4096',MOSS_REPEATS='0',MOSS_SEED='1',
          MOSS_MTP='3',MOSS_STAGE_FENCES='1',MOSS_FOUR_STOP_AFTER_FIRST='1',MOSS_FOUR_RAW_INPUT='1')
cases=[('state','5'),('qkv','12'),('weights','1,2,10'),('other','0,3,4,9,13,14,15'),
       ('control','15'),('side-buffers','13,14'),('activations','0,9'),('decay','3,4'),('input-x','0'),('input-z','9')]
selected=os.environ.get('MOSS_FOUR_INPUT_GROUPS','state,qkv,weights,other').split(',')
for name,indices in cases:
    if name not in selected:
        continue
    with (root/('four-input-'+name+'.log')).open('w') as log:
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('moss_mtp_four_capture.py'))],
                              env=dict(base,MOSS_RESULT_DIR=str(root/('four-input-'+name)),MOSS_FOUR_CANONICALIZE=indices),stdout=log,stderr=subprocess.STDOUT)
    print(name,result.returncode,flush=True)
