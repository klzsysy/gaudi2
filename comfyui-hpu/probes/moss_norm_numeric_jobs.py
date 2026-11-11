"""Real Q/K normalization and a nested-compilation control."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ['MOSS_COMPLETION_ROOT'])
probes = Path(__file__).parent
base = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so', PYTHONPATH=str(Path.cwd()),
            MOSS_WORKFLOW=str(root/'baseline/source.json'), MOSS_GDN_MODE='product', MOSS_SEED='1',
            MOSS_CAPTION=str(root.parent/'performance-2026-11-10-dit-next/caption.txt'))
jobs = [
    ('norm-qk-reference', 'moss_norm_reference.py', dict(MOSS_NORM_QK='1', MOSS_LENGTH='8', MOSS_REPEATS='0')),
    ('mtp-outer-norm', 'moss_mtp_program.py', dict(MOSS_FINAL_REGION='outer_norm', MOSS_LENGTH='4096', MOSS_REPEATS='5', MOSS_MTP_PROGRAM=str(root/'mtp-record-298/program.pt'))),
    ('mtp-outer-norm-logits', 'moss_mtp_program.py', dict(MOSS_FINAL_REGION='outer_norm', MOSS_MTP_LOGITS='1', MOSS_LENGTH='4096', MOSS_REPEATS='1', MOSS_MTP_PROGRAM=str(root/'mtp-record-298/program.pt'))),
]
with (root/'norm-tests-weight-dtypes.log').open('w') as log:
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests-unit/comfy_test/hpu_qwen_norm_test.py'], env=base, stdout=log, stderr=subprocess.STDOUT)
print('weight dtype tests', result.returncode, flush=True)
for name,script,settings in jobs:
    with (root/(name+'.log')).open('w') as log:
        result = subprocess.run([sys.executable,str(probes/script)], env=dict(base,**settings,MOSS_RESULT_DIR=str(root/name)), stdout=log, stderr=subprocess.STDOUT)
    print(name,result.returncode,flush=True)
    summary = root/name/'summary.json'
    if summary.exists():
        print(summary.read_text(),flush=True)
