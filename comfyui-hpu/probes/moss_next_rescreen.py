"""Recheck compiler choices after changing the recurrence graph."""
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(os.environ['MOSS_NEXT_ROOT'])
probes=Path(__file__).parent
base=dict(os.environ,PT_HPU_LAZY_MODE='0',GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',PYTHONPATH=str(Path.cwd()),
          MOSS_WORKFLOW=str(root/'source.json'),MOSS_GDN_MODE='product',MOSS_LENGTH='4096',MOSS_REPEATS='5',MOSS_SEED='1',
          MOSS_MTP_PROGRAM=str(root.parent/'performance-2026-11-11-completion/mtp-record-298/program.pt'))
with (root/'delta-compile-compare.log').open('w') as log:
    subprocess.run([sys.executable,str(probes/'moss_delta_compile_compare.py')],env=base,stdout=log,stderr=subprocess.STDOUT,check=True)
micro=json.loads((root/'delta-compile-compare.json').read_text())
cases=[('parallel-region',dict(MOSS_DECODE_REGIONS='delta')),('parallel-repeat',{})]
if any(r['median_ms']<.98*next(p['median_ms'] for p in micro if p['capture']==r['capture'] and p['name']=='product')
       for r in micro if r['name']=='product-static'):
    cases.insert(0,('parallel-static',dict(MOSS_DELTA_STATIC='1')))
for name,settings in cases:
    print(name,'START',flush=True)
    with (root/(name+'.log')).open('w') as log:
        subprocess.run([sys.executable,str(probes/'moss_mtp_program.py')],env=dict(base,**settings,MOSS_RESULT_DIR=str(root/name)),
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    print(name,(root/name/'summary.json').read_text(),flush=True)
