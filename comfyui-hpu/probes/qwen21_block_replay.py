"""Replay captured Qwen blocks, with sampled CPU FP64 attention/FFN reference."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time
import torch
import torch.nn.functional as F
from comfy.cli_args import args
args.disable_dynamic_vram=True
args.disable_pinned_memory=True
args.disable_async_offload=True
args.use_pytorch_cross_attention=True
import comfy.ops
from comfy.ldm.qwen_image21 import model as dit

p=argparse.ArgumentParser()
p.add_argument('capture',type=Path)
p.add_argument('output',type=Path)
p.add_argument('--trial',default='')
p.add_argument('--reference',type=Path)
p.add_argument('--profile',action='store_true')
p.add_argument('--qkv-audit',action='store_true')
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False)
torch.set_num_threads(32)
saved=torch.load(a.capture,weights_only=True)
state=saved['state']
dim=state['attn.to_q.weight'].shape[0]
heads=saved['heads']
block=dit.QwenImage21TransformerBlock(dim,heads,dim//heads,mlp_ratio=state['img_mlp.out.weight'].shape[1]//dim,
                                    fused_mlp=True,dtype=torch.bfloat16,device='cpu',operations=comfy.ops.manual_cast)
block.load_state_dict(state)
block.to('hpu')
x=saved['x'].to('hpu')
mod=tuple(tuple(v.to('hpu') for v in item) if isinstance(item,tuple) else item.to('hpu') for item in saved['mod'])
pe=saved['pe'].to('hpu')
if a.trial:
    from qwen21_deep_candidates import install
    install(a.trial)
attn=dit.prefix_cached_attention(saved['prefix_k'].to('hpu'),saved['prefix_v'].to('hpu'))
work=x.clone()
def run():
    work.copy_(x)
    return block(work,mod,pe,attn,saved['prefix_len'])
start=time.perf_counter();actual=run();torch.hpu.synchronize()
report={'trial':a.trial,'shape':list(x.shape),'cold':time.perf_counter()-start}
actual=actual.cpu().double()
for _ in range(5):run()
torch.hpu.synchronize()
timings=[]
for _ in range(5):
    start=time.perf_counter()
    for _ in range(10):run()
    torch.hpu.synchronize();timings.append((time.perf_counter()-start)/10)
report.update(seconds=timings,median=statistics.median(timings))
rows=torch.linspace(0,x.shape[1]-1,8).long()
if a.reference:
    reference=torch.load(a.reference,weights_only=True)
    expected=reference['actual'].double()
    precise=reference['precise']
else:
    # Full keys/values and sampled queries retain the original attention domain.
    full=saved['x'].double()
    scale1,gate1,scale2,gate2,_=saved['mod']
    normalized=F.layer_norm(full,(dim,),eps=1e-6)*(1+scale1[1].double())
    def linear(t,key):return F.linear(t,state[key].double())
    q=linear(normalized[:,rows],'attn.to_q.weight').view(1,8,heads,-1)
    k=linear(normalized,'attn.to_k.weight').view(1,-1,heads,dim//heads)
    v=linear(normalized,'attn.to_v.weight').view_as(k)
    def rms(t,weight):return t*torch.rsqrt(t.square().mean(-1,keepdim=True)+1e-6)*state[weight].double()
    def rope(t,frequencies):
        pairs=t.reshape(*t.shape[:-1],-1,1,2)
        return (frequencies[...,0]*pairs[...,0]+frequencies[...,1]*pairs[...,1]).reshape_as(t)
    q=rope(rms(q,'attn.norm_q.weight'),saved['pe'][:,rows].double())
    k=rope(rms(k,'attn.norm_k.weight'),saved['pe'].double())
    k=torch.cat([saved['prefix_k'].double(),k],dim=1)
    v=torch.cat([saved['prefix_v'].double(),v],dim=1)
    q,k,v=(t.transpose(1,2) for t in (q,k,v))
    probabilities=(q @ k.transpose(-1,-2) / (dim//heads)**.5).softmax(-1)
    attended=(probabilities @ v).transpose(1,2).flatten(2)
    hidden=full[:,rows]+linear(attended,'attn.to_out.0.weight')*gate1[1].double()
    normalized=F.layer_norm(hidden,(dim,),eps=1e-6)*(1+scale2[1].double())
    gate,up=linear(normalized,'img_mlp.gate_up.weight').chunk(2,-1)
    precise=hidden+linear(F.silu(gate)*up,'img_mlp.out.weight')*gate2[1].double()
    expected=actual
    torch.save({'actual':actual.to(torch.bfloat16),'precise':precise},a.output/'reference.pt')
report['rrmse_to_baseline']=float((actual-expected).square().mean().sqrt()/expected.square().mean().sqrt())
report['baseline_fp64_error']=float((expected[:,rows]-precise).square().mean().sqrt()/precise.square().mean().sqrt())
report['candidate_fp64_error']=float((actual[:,rows]-precise).square().mean().sqrt()/precise.square().mean().sqrt())
report['fp64_error_ratio']=report['candidate_fp64_error']/report['baseline_fp64_error']
report['finite']=bool(torch.isfinite(actual).all())
(a.output/'result.json').write_text(json.dumps(report,indent=2))
print(report,flush=True)
if a.profile:
    gc.collect();gc.disable()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.HPU]) as profile:
        run();torch.hpu.synchronize()
    profile.profiler.export_chrome_trace(str(a.output/'block.trace.json'))
    (a.output/'operators.txt').write_text(profile.key_averages().table(sort_by='self_cpu_time_total',row_limit=60))
    gc.enable()

if a.qkv_audit:
    from qwen21_deep_candidates import native_rope, functional_rope, qkv
    normalized = dit._modulated_norm(block.img_norm1, x, mod[0], 0, mod[-1])
    query = block.attn.to_q(normalized).view(1, x.shape[1], heads, -1)
    key = block.attn.to_k(normalized).view_as(query)
    q_scale = block.attn.norm_q.weight
    k_scale = block.attn.norm_k.weight
    q_norm = F.rms_norm(query,(query.shape[-1],),q_scale,1e-6)
    expected,_ = comfy.ops.quant_ops.ck.rms_rope(query,key,pe,q_scale,k_scale,1e-6)
    selected = rows.to('hpu')
    expected = expected.index_select(1,selected).cpu().double()
    norm_rows = q_norm.index_select(1,selected).cpu().double()
    frequencies = saved['pe'][:,rows].double()
    pairs = norm_rows.reshape(*norm_rows.shape[:-1],-1,1,2)
    reference = (frequencies[...,0]*pairs[...,0]+frequencies[...,1]*pairs[...,1]).reshape_as(norm_rows)
    audit={}
    for label, result in [('native_eager', native_rope(q_norm,pe)),('functional_eager',functional_rope(q_norm,pe))]:
        output=result.index_select(1,selected).cpu().double()
        audit[label]=dict(to_baseline=float((output-expected).square().mean().sqrt()/expected.square().mean().sqrt()),
                          to_fp64=float((output-reference).square().mean().sqrt()/reference.square().mean().sqrt()))
    audit['baseline_to_fp64']=float((expected-reference).square().mean().sqrt()/reference.square().mean().sqrt())
    (a.output/'qkv-audit.json').write_text(json.dumps(audit,indent=2))
    print(audit,flush=True)
