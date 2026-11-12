"""Serialize evidence jobs on the independent checkout and restore its service."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


root=Path(os.environ['MOSS_EVIDENCE_ROOT'])
probes=Path(__file__).parent
python=sys.executable
base=dict(os.environ,PT_HPU_LAZY_MODE='0',GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
          PYTHONPATH=str(Path.cwd()))
profile=dict(HABANA_PROFILE='1',HABANA_PROF_CONFIG=str(root/'inventory/counters.json'))
model='/home/sonny/ComfyUI-Models/models/diffusion_models/Moody-Moss-Mix-V4.1_00001__clean-fp8.safetensors'
capture=root.parent/'performance-2026-11-12-fp8-deep/baseline/captures/input-307-12-0.pt'
jobs=[
    ('raw-copy-corrected',[python,str(probes/'moss_evidence_hardware.py'),str(root/'raw-copy-corrected'),'--raw','copy'],profile),
    ('raw-mme',[python,str(probes/'moss_evidence_hardware.py'),str(root/'raw-mme'),'--raw','mme'],profile),
    ('raw-tpc',[python,str(probes/'moss_evidence_hardware.py'),str(root/'raw-tpc'),'--raw','tpc'],profile),
    ('block-counters',[python,str(probes/'moss_evidence_block.py'),str(root/'block-counters'),'--input',str(capture),'--model',model],
     dict(profile,DUMP_POST_GRAPHS=str(root/'block-counters/post'))),
    ('quality',[python,str(probes/'moss_evidence_quality.py')],{}),
    ('precision',[python,str(probes/'moss_evidence_precision.py')],{}),
    ('text',[python,str(probes/'moss_evidence_text.py')],dict(MOSS_RESULT_DIR=str(root/'text'),
         MOSS_MTP_PROGRAM=str(root.parent/'performance-2026-11-11-completion/mtp-record-298/program.pt'),
         MOSS_WORKFLOW=str(root/'dit-six-retry/source.json'),
         MOSS_CLIP_FILE='/home/sonny/ComfyUI-Models/models/text_encoders/qwen3.5_9b_fp8_from_int8.safetensors')),
]
if os.environ.get('MOSS_EVIDENCE_PHASE')=='2':
    jobs=[('power-control',[python,str(probes/'moss_evidence_power_control.py')],{})]
    jobs += [
        (f'native-{operation}',[python,str(probes/'moss_evidence_hardware.py'),str(root/f'native-{operation}'),
          '--raw',operation,'--native'],profile) for operation in ['copy','mme','tpc']]
    jobs += [
        ('block-native',[python,str(probes/'moss_evidence_block.py'),str(root/'block-native'),'--input',str(capture),
                         '--model',model,'--native'],dict(profile,DUMP_POST_GRAPHS=str(root/'block-native/post'))),
        ('dit-gc',[python,str(probes/'moss_fp8_deep_jobs.py')],dict(MOSS_DEEP_ROOT=str(root),MOSS_DEEP_CASE='dit-gc',
            MOSS_DEEP_PROFILE='1',MOSS_DEEP_PROFILE_STEPS='0,12,24',MOSS_PROFILE_GC_CONTROL='1')),
        ('text-gc',[python,str(probes/'moss_evidence_text.py')],dict(MOSS_RESULT_DIR=str(root/'text-gc'),
            MOSS_MTP_PROGRAM=str(root.parent/'performance-2026-11-11-completion/mtp-record-298/program.pt'),
            MOSS_WORKFLOW=str(root/'dit-six-retry/source.json'),MOSS_PROFILE_GC_CONTROL='1',MOSS_TEXT_PROFILE_REPEATS='1',
            MOSS_CLIP_FILE='/home/sonny/ComfyUI-Models/models/text_encoders/qwen3.5_9b_fp8_from_int8.safetensors'))]
if os.environ.get('MOSS_EVIDENCE_PHASE') in ('3','4'):
    jobs=[]
    suffix='-tef' if os.environ['MOSS_EVIDENCE_PHASE']=='4' else ''
    for operation in ['copy','mme','tpc']:
        name=f'counters-{operation}{suffix}'
        jobs.append((name,[python,str(probes/'moss_evidence_hardware.py'),str(root/name),
                          '--raw',operation,'--native'],profile.copy()))
    for label,condition in [('full',307),('negative',23)]:
        name=f'block-native-{label}{suffix}'
        source=capture.with_name(f'input-{condition}-12-0.pt')
        jobs.append((name,[python,str(probes/'moss_evidence_block.py'),str(root/name),'--input',str(source),
                          '--model',model,'--native'],dict(profile,DUMP_POST_GRAPHS=str(root/name/'post'))))
    for name,command,settings in jobs:
        directory=root/name
        directory.mkdir(exist_ok=True)
        configuration=json.loads((root/'inventory/counters.json').read_text())
        configuration['GeneralSettings']['values']['outdir']['value']=str(directory/'hw')
        configuration['GeneralSettings']['values']['session']['value']=name
        (directory/'hw').mkdir(exist_ok=True)
        (directory/'profiler.json').write_text(json.dumps(configuration,indent=2))
        settings['HABANA_PROF_CONFIG']=str(directory/'profiler.json')
if os.environ.get('MOSS_EVIDENCE_PHASE')=='5':
    jobs=[]
    for mib in [128,512]:
        name=f'hbm-copy-{mib}'
        jobs.append((name,[python,str(probes/'moss_evidence_hardware.py'),str(root/name),'--raw','copy',
                          '--native','--iterations','5','--mib',str(mib)],profile.copy()))
    for label,condition in [('positive',307),('negative',23)]:
        name=f'block-hardware-{label}'
        jobs.append((name,[python,str(probes/'moss_evidence_block.py'),str(root/name),'--input',
                          str(capture.with_name(f'input-{condition}-12-0.pt')),'--model',model,'--native'],profile.copy()))
    for name,command,settings in jobs:
        directory=root/name
        directory.mkdir(exist_ok=True)
        configuration=json.loads((root/'counter-calibration/profiler.json').read_text())
        configuration['GeneralSettings']['values']['outdir']['value']=str(directory/'hw')
        configuration['GeneralSettings']['values']['session']['value']=name
        (directory/'hw').mkdir(exist_ok=True)
        (directory/'profiler.json').write_text(json.dumps(configuration,indent=2))
        settings['HABANA_PROF_CONFIG']=str(directory/'profiler.json')
try:
    for name,command,settings in jobs:
        start=time.time()
        with (root/f'{name}.log').open('w') as log:
            process=subprocess.Popen(command,env=dict(base,**settings),stdout=log,stderr=subprocess.STDOUT)
            print(name,process.pid,flush=True)
            try:
                code=process.wait(timeout=2400)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                code='timeout'
        record=dict(name=name,pid=process.pid,returncode=code,seconds=time.time()-start)
        with (root/'jobs.jsonl').open('a') as output:
            output.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
finally:
    with (root/'restore-service.log').open('w') as log:
        subprocess.run(['./start-hpu.sh'],stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
