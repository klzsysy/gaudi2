"""Final regional-compilation acceptance with matched work and normal entry points."""
import os
from pathlib import Path
import subprocess
import sys

root=Path(os.environ['MOSS_NEXT_ROOT'])
probes=Path(__file__).parent
base=dict(os.environ,PT_HPU_LAZY_MODE='0',GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',PYTHONPATH=str(Path.cwd()),
          MOSS_WORKFLOW=str(root/'source.json'),MOSS_GDN_MODE='product',MOSS_LENGTH='4096',MOSS_REPEATS='5',MOSS_SEED='1',
          MOSS_MTP_PROGRAM=str(root.parent/'performance-2026-11-11-completion/mtp-record-298/program.pt'))
matched=root/'matched-final'
matched.mkdir(exist_ok=True)
(matched/'source.json').write_text((root/'source.json').read_text())
stages=[
 ('region-tests',['-m','pytest','tests-unit/comfy_test/hpu_gdn_region_test.py','-q'],{}),
 ('region-logits-baseline',[str(probes/'moss_mtp_program.py')],dict(MOSS_REGION_BASELINE='1',MOSS_MTP_LOGITS='1',MOSS_REPEATS='1',MOSS_RESULT_DIR=str(root/'region-logits-baseline'))),
 ('region-logits-candidate',[str(probes/'moss_mtp_program.py')],dict(MOSS_MTP_LOGITS='1',MOSS_REPEATS='1',MOSS_RESULT_DIR=str(root/'region-logits-candidate'))),
 ('region-logits-comparison',[str(probes/'moss_region_logits.py'),str(root)],{}),
 ('region-product-text',[str(probes/'moss_mtp_program.py')],dict(MOSS_RESULT_DIR=str(root/'region-product-text'))),
 ('region-product-profile',[str(probes/'moss_mtp_program.py')],dict(MOSS_MTP_PROFILE='1',MOSS_REPEATS='1',MOSS_RESULT_DIR=str(root/'region-product-profile'))),
 ('matched-final',[str(probes/'moss_pipeline_measure.py')],dict(MOSS_MTP_FIXED='1',MOSS_PIPELINE_DIR=str(matched),MOSS_RESULT_DIR=str(matched/'mtp'),MOSS_PIPELINE_CACHE_RUNS='0',MOSS_RELEASE='1')),
 ('quality-region-final',[str(probes/'moss_pipeline_measure.py')],dict(MOSS_PIPELINE_DIR=str(root/'natural-product'),MOSS_PIPELINE_FULL_RUNS='0',
    MOSS_PIPELINE_CACHE_RUNS='0',MOSS_PIPELINE_DIAGNOSTIC='0',MOSS_TREE_AB='1',MOSS_TREE_QUALITY='1',
    MOSS_QUALITY_LABELS='teapot-candidate,poster-candidate,person-candidate,mtp-off,mtp-3,mtp-tree3s1,int8',MOSS_RELEASE='1')),
 ('final-tests',[str(probes/'moss_next_acceptance.py')],dict(MOSS_NEXT_STAGES='tests,product-validation')),
 ('four-final',[str(probes/'moss_delta_four_replay.py')],dict(MOSS_FOUR_MODE='product',MOSS_FOUR_PARAMETERS='1')),
]
selected=os.environ.get('MOSS_REGION_STAGES',','.join(name for name,_,_ in stages)).split(',')
for name,command,settings in stages:
    if name not in selected:
        continue
    print(name,'START',flush=True)
    with (root/(name+'.log')).open('w') as log:
        subprocess.run([sys.executable,*command],env=dict(base,**settings),stdout=log,stderr=subprocess.STDOUT,check=True)
    print(name,'PASS',flush=True)
