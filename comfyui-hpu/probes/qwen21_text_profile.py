"""Profile Qwen PE decode on the main thread with matching cache capacity."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from comfy.cli_args import args
args.disable_dynamic_vram=True
args.disable_pinned_memory=True
args.disable_async_offload=True
args.use_pytorch_cross_attention=True
import comfy.sd
from comfy.text_encoders import qwen35, hpu_mlp
from comfy.text_encoders.llama import BaseGenerate

p=argparse.ArgumentParser()
p.add_argument('output',type=Path)
p.add_argument('--trial',default='')
p.add_argument('--teacher',type=Path)
p.add_argument('--logits',action='store_true')
p.add_argument('--workflow',type=Path)
p.add_argument('--plain-only',action='store_true')
p.add_argument('--plain-runs',type=int,default=5)
p.add_argument('--window-tokens',type=int)
p.add_argument('--cpu-profile',action='store_true')
p.add_argument('--quality-only',action='store_true')
p.add_argument('--native',action='store_true')
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
if a.native:
    from moss_native_profile import NativeProfile
if a.trial:
    from qwen21_deep_candidates import install
    install(a.trial)
source=json.loads((a.workflow or Path('/home/sonny/gaudi2/comfyui-hpu/workflows/qwen21-user-bf16-api.json')).read_text())
settings=source['459:471']['inputs']
torch.set_grad_enabled(False)
torch.set_num_threads(32)
clip=comfy.sd.load_clip(['/home/sonny/ComfyUI-Models/models/text_encoders/'+source['459:473']['inputs']['clip_name']],clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
tokens=clip.tokenize(settings['prompt'],system_prompt=source['459:475']['inputs']['value'],min_length=1,thinking=settings['thinking'])
clip.load_model(tokens)
active=None
finished=None
counter=0
mode='warm'
window_start=0
window_seconds=None
teacher=json.loads(a.teacher.read_text()) if a.teacher else None
saved_logits=[]
sample=BaseGenerate.sample_token

def sampled(self,*values,**kwargs):
    global active,finished,counter,window_start,window_seconds
    result=sample(self,*values,**kwargs)
    if teacher is not None:
        result=torch.full_like(result,teacher[counter])
    counter+=1
    if a.window_tokens and counter == a.window_tokens:
        result = torch.full_like(result, self.model.config.stop_tokens[0])
    if counter==(512 if a.native else 32):
        torch.hpu.synchronize()
        if mode=='profile':
            active=NativeProfile(a.output) if a.native else torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.HPU])
            active.__enter__()
        if mode=='cpu_profile':
            cpu_profile.enable()
        window_start=time.perf_counter()
    if counter==(528 if a.native else 48):
        torch.hpu.synchronize()
        window_seconds=time.perf_counter()-window_start
        if mode=='cpu_profile':
            cpu_profile.disable()
        if active:
            active.__exit__(None,None,None)
            finished,active=None if a.native else active,None
        if a.native and mode in ('plain','profile'):
            (a.output/('window.json' if mode=='profile' else 'plain-window.json')).write_text(
                json.dumps(dict(seconds=window_seconds,inter_token_intervals=16,forwards=16)))
    return result
BaseGenerate.sample_token=sampled

def labelled(function,label):
    def call(*values,**kwargs):
        if not a.native and active:
            with torch.profiler.record_function(label):return function(*values,**kwargs)
        return function(*values,**kwargs)
    return call
qwen35.Qwen35Transformer.forward=labelled(qwen35.Qwen35Transformer.forward,'PE_transformer')
qwen35.GatedDeltaNet.forward=labelled(qwen35.GatedDeltaNet.forward,'PE_GDN')
qwen35.GatedAttention.forward=labelled(qwen35.GatedAttention.forward,'PE_attention')
hpu_mlp.forward=labelled(hpu_mlp.forward,'PE_MLP')
BaseGenerate.logits=labelled(BaseGenerate.logits,'PE_head')
BaseGenerate.processed_probs=labelled(BaseGenerate.processed_probs,'PE_probabilities')
head=BaseGenerate.logits
def logits(self,*values,**kwargs):
    result=head(self,*values,**kwargs)
    if mode=='logits':saved_logits.append(result[:,-1].detach().cpu().clone())
    return result
BaseGenerate.logits=logits
sequence=['warm','plain','plain','plain','logits'] if a.logits else ['warm','plain','profile','plain']
if a.plain_only:
    sequence=['warm']+['plain']*a.plain_runs
if a.quality_only:
    sequence=['warm','logits']
if a.cpu_profile:
    import cProfile
    cpu_profile=cProfile.Profile()
    sequence=['warm','plain','cpu_profile','plain']
for mode in sequence:
    counter=0
    finished=None
    gc.collect();gc.disable()
    torch.hpu.synchronize();start=time.perf_counter()
    ids=clip.generate(tokens,do_sample=True,max_length=settings['max_length'],temperature=1.0,top_k=20,top_p=.95,min_p=0.,repetition_penalty=1.,presence_penalty=1.5,seed=settings['sampling_mode.seed'],mtp=True)
    torch.hpu.synchronize()
    result=dict(mode=mode,seconds=time.perf_counter()-start,window_seconds=window_seconds,tokens=len(ids))
    with (a.output/'timings.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    print(result,flush=True)
    (a.output/f'{mode}-tokens.json').write_text(json.dumps(ids))
    if finished:
        finished.profiler.export_chrome_trace(str(a.output/'text.trace.json'))
        (a.output/'operators.txt').write_text(finished.key_averages().table(sort_by='self_cpu_time_total',row_limit=60))
    gc.enable()

if saved_logits:torch.save(torch.cat(saved_logits),a.output/'logits.pt')
if a.cpu_profile:
    cpu_profile.dump_stats(str(a.output/'cpu-profile.pstats'))

torch.hpu.synchronize()
torch._dynamo.reset()
comfy.model_management.unload_all_models()
del clip
gc.collect()
