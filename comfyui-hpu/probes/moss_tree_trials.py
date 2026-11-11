"""Serial follow-up measurements for the parallel recurrence prototype."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ['MOSS_NEXT_ROOT'])
base = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
            PYTHONPATH=str(Path.cwd()), MOSS_WORKFLOW=str(root/'source.json'), MOSS_GDN_MODE='product',
            MOSS_LENGTH='4096', MOSS_REPEATS='5', MOSS_SEED='1',
            MOSS_MTP_PROGRAM=str(root.parent/'performance-2026-11-11-completion/mtp-record-298/program.pt'))
cases = [
    ('conv-control', dict(MOSS_CONV_CONTROL='1')),
    ('tree-conv', dict(MOSS_TREE_DELTA='1', MOSS_CONV_CONTROL='1')),
    ('baseline-capture', dict(MOSS_DELTA_CAPTURE='1', MOSS_MTP_LOGITS='1', MOSS_REPEATS='0')),
    ('tree-logits', dict(MOSS_TREE_DELTA='1', MOSS_MTP_LOGITS='1', MOSS_REPEATS='0')),
]
for name, settings in cases:
    directory = root/name
    with (root/(name+'.log')).open('w') as log:
        result = subprocess.run([sys.executable,str(Path(__file__).with_name('moss_mtp_program.py'))],
                                env=dict(base,**dict(settings,MOSS_RESULT_DIR=str(directory))), stdout=log,stderr=subprocess.STDOUT)
    print(name,result.returncode,flush=True)
    if (directory/'summary.json').exists():
        print((directory/'summary.json').read_text(),flush=True)
