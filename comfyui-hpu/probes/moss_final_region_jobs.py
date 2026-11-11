"""Serial final text-region screen against the accepted norm-only candidate."""
import json
import os
from pathlib import Path
import subprocess
import sys
root=Path(os.environ['MOSS_COMPLETION_ROOT'])
probes=Path(__file__).parent
base=dict(os.environ,PT_HPU_LAZY_MODE='0',GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',PYTHONPATH=str(Path.cwd()),MOSS_MTP_PROGRAM=str(root/'mtp-record-298/program.pt'),MOSS_WORKFLOW=str(root/'baseline/source.json'),MOSS_GDN_MODE='product',MOSS_LENGTH='4096',MOSS_REPEATS='5',MOSS_SEED='1')
for mode in ['block','mtp','probability']:
    name='final-region-'+mode
    with (root/(name+'.log')).open('w') as log:
        result=subprocess.run([sys.executable,str(probes/'moss_mtp_program.py')],env=dict(base,MOSS_FINAL_REGION=mode,MOSS_RESULT_DIR=str(root/name)),stdout=log,stderr=subprocess.STDOUT)
    print(mode,result.returncode,flush=True)
    p=root/name/'summary.json'
    if p.exists():print(p.read_text(),flush=True)
