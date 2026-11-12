"""Compare fixed-MTP text windows with and without synchronization/profiling."""
import json
import gc
import os
from pathlib import Path
import time

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True
import comfy.sd
from comfy.text_encoders import qwen35
from comfy.text_encoders.llama import BaseGenerate
import habana_frameworks.torch.utils.debug as debug
import moss_mtp_program as program


root=Path(os.environ['MOSS_RESULT_DIR'])
source=json.loads(Path(os.environ['MOSS_WORKFLOW']).read_text())
torch.set_grad_enabled(False)
torch.set_num_threads(32)
clip=comfy.sd.load_clip([os.environ['MOSS_CLIP_FILE']],clip_type=comfy.sd.CLIPType.MOSS_IMAGE)
tokens=clip.tokenize(source['18:17']['inputs']['value'],system_prompt=source['18:14']['inputs']['value'],
                     min_length=1,thinking=False)
clip.load_model(tokens)
mode='warmup'
active_profile=None
finished_profile=None
window_start=0
window_seconds=None
window_end_count=10
sample=BaseGenerate.sample_token


def observed_sample(self,*values,**kwargs):
    global active_profile,finished_profile,window_start,window_seconds
    result=sample(self,*values,**kwargs)
    index=program.indices['sample']
    if index==8:
        if mode!='plain':
            torch.hpu.synchronize()
        if mode.startswith('profile'):
            active_profile=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                             torch.profiler.ProfilerActivity.HPU])
            active_profile.__enter__()
        window_start=time.perf_counter_ns()
    if index==window_end_count:
        if mode!='plain':
            torch.hpu.synchronize()
        window_seconds=(time.perf_counter_ns()-window_start)/1e9
        if active_profile:
            active_profile.__exit__(None,None,None)
            finished_profile,active_profile=active_profile,None
    return result


BaseGenerate.sample_token=observed_sample


def labelled(function,label):
    def wrapped(*values,**kwargs):
        if active_profile:
            with torch.profiler.record_function(label):
                return function(*values,**kwargs)
        return function(*values,**kwargs)
    return wrapped


qwen35.Qwen35Transformer.forward=labelled(qwen35.Qwen35Transformer.forward,'evidence_text_main')
qwen35.MTPHead.forward=labelled(qwen35.MTPHead.forward,'evidence_mtp_head')
for name,module in clip.cond_stage_model.named_modules():
    if name.endswith('.lm_head'):
        module.forward=labelled(module.forward,'evidence_lm_head')

sequence=['warmup']+['plain']*int(os.environ.get('MOSS_TEXT_PLAIN_REPEATS','5'))+['sync2','profile2','plain','sync16','profile16','plain']*int(os.environ.get('MOSS_TEXT_PROFILE_REPEATS','3'))
for index,mode in enumerate(sequence):
    control_gc=os.environ.get('MOSS_PROFILE_GC_CONTROL')=='1'
    gc_enabled=gc.isenabled()
    if control_gc:
        gc.collect()
        gc.disable()
    window_end_count=24 if mode.endswith('16') else 10
    finished_profile=None
    torch.hpu.synchronize()
    start=time.perf_counter_ns()
    ids=clip.generate(tokens,max_length=4096,do_sample=True,temperature=.7,top_k=64,top_p=.95,min_p=.05,seed=0,mtp=True)
    torch.hpu.synchronize()
    record=dict(index=index,mode=mode,seconds=(time.perf_counter_ns()-start)/1e9,window_seconds=window_seconds,
                window_rounds=window_end_count-8,tokens=len(ids),fallback=debug._get_fallback_op_count())
    with (root/'perturbation.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)
    if finished_profile:
        finished_profile.profiler.export_chrome_trace(str(root/f'{index}-{mode}.trace.json'))
        finished_profile=None
    if control_gc and gc_enabled:
        gc.enable()
