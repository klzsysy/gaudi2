"""Serial fixed-MTP text projection trials."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ['MOSS_NEXT_ROOT'])
probes = Path(__file__).parent
base = dict(os.environ, PT_HPU_LAZY_MODE='0',GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',PYTHONPATH=str(Path.cwd()),
            MOSS_WORKFLOW=str(root/'source.json'),MOSS_GDN_MODE='product',MOSS_LENGTH='4096',MOSS_REPEATS='5',MOSS_SEED='1',
            MOSS_MTP_PROGRAM=str(root.parent/'performance-2026-11-11-completion/mtp-record-298/program.pt'))
for mode in ['baseline','delta','head','all']:
    directory = root/('linear-'+mode)
    settings = {} if mode=='baseline' else {'MOSS_LINEAR_REGION':mode}
    with (root/('linear-'+mode+'.log')).open('w') as log:
        result = subprocess.run([sys.executable,str(probes/'moss_mtp_program.py')],env=dict(base,**settings,MOSS_RESULT_DIR=str(directory)),stdout=log,stderr=subprocess.STDOUT)
    print(mode,result.returncode,flush=True)
    if (directory/'summary.json').exists():
        print((directory/'summary.json').read_text(),flush=True)
