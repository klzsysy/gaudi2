"""Serial remaining text experiments; no competing HPU process."""
import json
import os
from pathlib import Path
import subprocess
import sys

probes = Path(__file__).parent
root = Path(os.environ['MOSS_COMPLETION_ROOT'])
base = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so', PYTHONPATH=str(Path.cwd()),
            MOSS_WORKFLOW=str(root.parent/'performance-2026-11-11-bf16/baseline/source.json'), MOSS_GDN_MODE='product', MOSS_SEED='1',
            MOSS_CAPTION=str(root.parent/'performance-2026-11-10-dit-next/caption.txt'))
jobs = [('mtp-norm-profile', 'moss_mtp_program.py', dict(MOSS_MTP_PROFILE='1', MOSS_LENGTH='4096', MOSS_REPEATS='1', MOSS_MTP_PROGRAM=str(root/'mtp-record-298/program.pt')))]
for mode in ['product', 'mlp', 'attention', 'both']:
    jobs.append(('prefill-'+mode, 'moss_text_prefill_screen.py', dict(MOSS_LENGTH='1', MOSS_REPEATS='5', MOSS_PREFILL_SCREEN=mode)))
for name, script, settings in jobs:
    env = dict(base, **settings, MOSS_RESULT_DIR=str(root/name))
    with (root/(name+'.log')).open('w') as log:
        result = subprocess.run([sys.executable, str(probes/script)], env=env, stdout=log, stderr=subprocess.STDOUT)
    print(name, result.returncode, flush=True)
    if result.returncode == 0:
        for file in ['summary.json', 'encoding-summary.json']:
            path = root/name/file
            if path.exists():
                print(path.read_text(), flush=True)
