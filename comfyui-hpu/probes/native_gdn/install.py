"""Keep the existing deferred buffers and replace only the FP32 state core."""
import torch
import torch.nn.functional as F
import comfy_kitchen
from comfy.text_encoders import hpu_gated_delta as delta
from native_gdn.ops import core


def decode(x,w_a,w_b,dt_bias,g_decay,state,key_dim,num_key_heads,scale,
           z,norm_weight,eps,qkv_buf,gates_buf,sumsq_buf,ctl):
    batch,seq,_=x.shape
    heads,key_width,value_width=state.shape[1:]
    groups=heads//num_key_heads
    slots_max=comfy_kitchen.gated_delta_slot_max
    control=ctl.long()
    pending,parity=control[0],control[1]
    slots=control[2:2+slots_max]
    program=control[2+2*slots_max:2+3*slots_max]
    beta=F.linear(x,w_b).sigmoid().float()
    decay=torch.exp(g_decay*F.softplus(F.linear(x,w_a).float()+dt_bias))
    current=delta._slab(qkv_buf,parity)[:,:,:seq]
    query=current[:,:key_dim].reshape(batch,num_key_heads,key_width,seq).float()
    key=current[:,key_dim:2*key_dim].reshape(batch,num_key_heads,key_width,seq).float()
    value=current[:,2*key_dim:].reshape(batch,heads,value_width,seq).float()
    sumsq=torch.stack([query.square().sum(2),key.square().sum(2)],dim=-1)
    gates_buf.index_copy_(0,parity.view(1),F.pad(torch.stack([decay,beta],dim=-1),(0,0,0,0,0,slots_max-seq)).unsqueeze(0))
    sumsq_buf.index_copy_(0,parity.view(1),F.pad(sumsq.transpose(1,2),(0,0,0,0,0,slots_max-seq)).unsqueeze(0))
    live=pending>torch.arange(slots_max,device=ctl.device)
    previous=torch.where(live.view(1,1,slots_max),delta._slab(qkv_buf,1-parity).index_select(2,slots),0.0)
    previous_key=previous[:,key_dim:2*key_dim].reshape(batch,num_key_heads,key_width,slots_max).float()
    previous_value=previous[:,2*key_dim:].reshape(batch,heads,value_width,slots_max).float()
    previous_gates=delta._slab(gates_buf,1-parity).index_select(1,slots)
    previous_sumsq=torch.where(live.view(1,1,slots_max,1),delta._slab(sumsq_buf,1-parity).index_select(1,slots).transpose(1,2),1.0)
    previous_decay=torch.where(live.view(1,slots_max,1),previous_gates[...,0],1.0)
    previous_beta=torch.where(live.view(1,slots_max,1),previous_gates[...,1],0.0)
    previous_key=(previous_key/previous_sumsq[...,1].sqrt().clamp(min=1e-12).unsqueeze(2)).repeat_interleave(groups,dim=1)
    norms=sumsq.sqrt().clamp(min=1e-12).unsqueeze(2)
    query=(query/norms[...,0]*scale).repeat_interleave(groups,dim=1)
    key=(key/norms[...,1]).repeat_interleave(groups,dim=1)
    committed,output=core(state,previous_key,previous_value,previous_decay,previous_beta,
                          query,key,value,decay,beta,program[:seq],pending)
    state.copy_(committed)
    output=output.to(x.dtype).float()
    rstd=torch.rsqrt(output.square().mean(-1,keepdim=True)+eps)
    normalized=(output*rstd*norm_weight.float()).to(x.dtype)
    return normalized*F.silu(z.reshape(batch,seq,heads,value_width))


delta._decode_deferred=decode
delta.compiled_ops.cache_clear()
