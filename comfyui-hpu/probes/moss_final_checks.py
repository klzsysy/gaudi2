"""Existing vendor RMSNorm control followed by the final product regressions."""
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
    ('mtp-native-norm', 'moss_mtp_program.py', dict(MOSS_FINAL_REGION='native_norm', MOSS_LENGTH='4096', MOSS_REPEATS='5', MOSS_MTP_PROGRAM=str(root/'mtp-record-298/program.pt'))),
    ('native-norm-reference', 'moss_norm_reference.py', dict(MOSS_NORM_KERNEL='native', MOSS_LENGTH='8', MOSS_REPEATS='0')),
    ('native-norm-qk-reference', 'moss_norm_reference.py', dict(MOSS_NORM_KERNEL='native', MOSS_NORM_QK='1', MOSS_LENGTH='8', MOSS_REPEATS='0')),
]
for name,script,settings in jobs:
    with (root/(name+'.log')).open('w') as log:
        result = subprocess.run([sys.executable,str(probes/script)], env=dict(base,**settings,MOSS_RESULT_DIR=str(root/name)), stdout=log, stderr=subprocess.STDOUT)
    print(name,result.returncode,flush=True)
    summary = root/name/'summary.json'
    if summary.exists():
        print(summary.read_text(),flush=True)
destination = root/'matched-threads16'
destination.mkdir(exist_ok=True)
(destination/'source.json').write_text((root/'baseline/source.json').read_text())
env = dict(base, OMP_NUM_THREADS='16', MKL_NUM_THREADS='16', MOSS_PIPELINE_DIR=str(destination),
           MOSS_RESULT_DIR=str(destination/'mtp'), MOSS_MTP_FIXED='1',
           MOSS_MTP_PROGRAM=str(root/'mtp-record-298/program.pt'), MOSS_PIPELINE_CACHE_RUNS='0')
with (root/'matched-threads16.log').open('w') as log:
    subprocess.run([sys.executable,str(probes/'moss_pipeline_measure.py')], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
print('matched pipeline with 16 CPU threads PASS',flush=True)
tests = ['hpu_qwen_norm_test.py','hpu_mlp_test.py','qwen35_decode_test.py','hpu_gated_delta_test.py','hpu_gdn_prefill_test.py','hpu_kv_attention_test.py','hpu_textgen_test.py']
with (root/'final-text-tests.log').open('w') as log:
    subprocess.run([sys.executable,'-m','pytest','-q',*[str(Path('tests-unit/comfy_test')/name) for name in tests]], env=base, stdout=log, stderr=subprocess.STDOUT, check=True)
print('final text regressions PASS',flush=True)
