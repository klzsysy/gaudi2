"""Serial acceptance, stopping immediately on a failed stage."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ['MOSS_NEXT_ROOT'])
probes = Path(__file__).parent
base = dict(os.environ,PT_HPU_LAZY_MODE='0',GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',PYTHONPATH=str(Path.cwd()),
            MOSS_WORKFLOW=str(root/'source.json'),MOSS_GDN_MODE='product',MOSS_LENGTH='4096',MOSS_REPEATS='5',MOSS_SEED='1',
            MOSS_MTP_PROGRAM=str(root.parent/'performance-2026-11-11-completion/mtp-record-298/program.pt'))
tests = ['hpu_gated_delta_test.py','qwen35_decode_test.py','hpu_qwen_norm_test.py','hpu_textgen_test.py',
         'hpu_gdn_prefill_test.py','hpu_kv_attention_test.py','hpu_mlp_test.py','hpu_moss_attention_test.py','hpu_moss_block_test.py','hpu_gdn_region_test.py']
natural = root/'natural-product'
natural.mkdir(exist_ok=True)
(natural/'source.json').write_text((root/'source.json').read_text())
stages = [
    ('tests',['-m','pytest',*[str(Path('tests-unit/comfy_test')/p) for p in tests],'-q'],{}),
    ('product-validation',[str(probes/'moss_tree_validate.py')],dict(MOSS_VALIDATE_PRODUCT='1')),
    ('head-product',[str(probes/'moss_head_product_checks.py')],{}),
    ('product-text',[str(probes/'moss_mtp_program.py')],dict(MOSS_RESULT_DIR=str(root/'product-text'))),
    ('product-profile',[str(probes/'moss_mtp_program.py')],dict(MOSS_RESULT_DIR=str(root/'product-profile'),MOSS_MTP_PROFILE='1',MOSS_REPEATS='1')),
    ('matched-pipeline',[str(probes/'moss_next_pipeline_jobs.py')],{}),
    ('natural-pipeline',[str(probes/'moss_pipeline_measure.py')],dict(MOSS_PIPELINE_DIR=str(natural),MOSS_TREE_AB='1',
         MOSS_TREE_QUALITY='1',MOSS_RELEASE='1',MOSS_PIPELINE_CACHE_RUNS='2')),
]
selected = os.environ.get('MOSS_NEXT_STAGES',','.join(name for name,_,_ in stages)).split(',')
for name,command,settings in stages:
    if name not in selected:
        continue
    print(name,'START',flush=True)
    with (root/(name+'.log')).open('w') as log:
        subprocess.run([sys.executable,*command],env=dict(base,**settings),stdout=log,stderr=subprocess.STDOUT,check=True)
    print(name,'PASS',flush=True)
