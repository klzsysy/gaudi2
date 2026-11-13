"""Process-local BF16 candidates. No changes to checkpoint precision or schedules."""
import torch
import comfy.ops
import comfy.model_management
from comfy.ldm.qwen_image21 import model as dit
from comfy.text_encoders import qwen35, llama
from comfy.text_encoders.llama import BaseGenerate
from comfy_kitchen.backends.eager.rope import apply_rope1


def native_rope(x, pe):
    cos = torch.stack((pe[...,0,0], pe[...,1,1]), dim=-1).flatten(-2)
    sin = torch.stack((-pe[...,0,1], pe[...,1,0]), dim=-1).flatten(-2)
    return torch.ops.hpu.rotary_pos_embedding(x.to(pe.dtype), sin, cos, None, 0, 1).to(x.dtype)


def functional_rope(x, pe):
    pairs = x.to(pe.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    products = pe[..., 0] * pairs[..., 0]
    return torch.addcmul(products, pe[..., 1], pairs[..., 1]).reshape_as(x).to(x.dtype)


def qkv(module, x, pe, native=False, fp32_norm=False, functional=False):
    batch, length, _ = x.shape
    q = module.to_q(x).view(batch, length, module.heads, -1)
    k = module.to_k(x).view(batch, length, module.heads, -1)
    v = module.to_v(x).view(batch, length, module.heads, -1)
    q_scale, _, q_stream = comfy.ops.cast_bias_weight(module.norm_q, q, offloadable=True)
    k_scale, _, k_stream = comfy.ops.cast_bias_weight(module.norm_k, k, offloadable=True)
    if native:
        if fp32_norm:
            q = torch.ops.hpu.rms_norm(q.float(), q_scale.float(), module.norm_q.eps)[0].to(q.dtype)
            k = torch.ops.hpu.rms_norm(k.float(), k_scale.float(), module.norm_k.eps)[0].to(k.dtype)
        else:
            q = torch.nn.functional.rms_norm(q, (q.shape[-1],), q_scale, module.norm_q.eps)
            k = torch.nn.functional.rms_norm(k, (k.shape[-1],), k_scale, module.norm_k.eps)
        rotate = functional_rope if functional else native_rope
        q, k = rotate(q, pe), rotate(k, pe)
    else:
        q, k = comfy.ops.quant_ops.ck.rms_rope(q, k, pe, q_scale, k_scale, module.norm_q.eps)
    comfy.ops.uncast_bias_weight(module.norm_q, q_scale, None, q_stream)
    comfy.ops.uncast_bias_weight(module.norm_k, k_scale, None, k_stream)
    return q, k, v


def install(mode):
    global native_rope
    parts = mode.split('+')
    if 'pairwise' in parts:
        native_rope = torch.compile(apply_rope1, backend='hpu_backend', dynamic=None, options={'force_static_compile':True,'decomposition_mode':'none'})
    if 'mask' in parts:
        original_local_attention = dit.attention
        def mask_eager(q,k,v,heads,mask=None,transformer_options={},preferred_attention=None):
            if mask is not None:
                return dit.optimized_attention(q,k,v,heads,mask=mask,transformer_options=transformer_options,preferred_attention=preferred_attention)
            return original_local_attention(q,k,v,heads,mask=mask,transformer_options=transformer_options,preferred_attention=preferred_attention)
        dit.attention = mask_eager
    if 'eager_attention' in parts:
        dit._compiled_attention = lambda: dit._attention
    if 'eager_rope' in parts:
        dit._hpu_rms_rope = comfy.ops.quant_ops.ck.rms_rope
    if 'baseline' in parts:
        dit._hpu_rms_rope = comfy.ops.quant_ops.ck.rms_rope
        dit._compiled_attention = lambda: dit._attention
        llama._compiled_processed_probs = lambda: BaseGenerate._processed_probs
    if 'delta1' in parts or 'delta1none' in parts:
        original = qwen35.GatedDeltaNet.forward
        delta_compiled = torch.compile(qwen35.GatedDeltaNet._forward, backend='hpu_backend', dynamic=None, options={'force_static_compile':True,'decomposition_mode':'none'}) if 'delta1none' in parts else qwen35._compiled_hpu_delta()
        def forward(self, x, past_key_value=None, **kwargs):
            if (x.device.type=='hpu' and x.dtype==torch.bfloat16 and x.shape[1]==1
                    and type(past_key_value) is qwen35.LinearKV and past_key_value.decode_step(1)):
                return delta_compiled(self,x,past_key_value=past_key_value,**kwargs)
            return original(self,x,past_key_value=past_key_value,**kwargs)
        qwen35.GatedDeltaNet.forward = forward
    if 'probability' in parts:
        BaseGenerate.processed_probs = torch.compile(BaseGenerate.processed_probs,backend='hpu_backend',dynamic=None,options={'force_static_compile':True})
    if 'qkv' in parts or 'rope' in parts or 'rope32' in parts or 'functional' in parts or 'splitrope' in parts or 'pairwise' in parts:
        qkv_options = {'force_static_compile':True}
        if 'functional' in parts:qkv_options['decomposition_mode'] = 'none'
        compiled_qkv = qkv if 'splitrope' in parts or 'pairwise' in parts else torch.compile(qkv,backend='hpu_backend',dynamic=None,options=qkv_options)
        original_attention = dit.Attention.forward
        def projected(self,x,pe,attn_fn,prefix_len,transformer_options={}):
            if (x.device.type=='hpu' and x.dtype==torch.bfloat16 and not transformer_options.get('patches')
                    and not comfy.ops.has_forward_hooks(self)):
                q,k,v = compiled_qkv(self,x,pe,native=any(p in parts for p in ('rope','rope32','functional','splitrope','pairwise')),fp32_norm='rope32' in parts,functional='functional' in parts)
                tensors=(dit.AttentionTensorContainer(t) for t in (q,k,v))
                return self.to_out[0](attn_fn(*tensors,self.heads,preferred_attention=self.comfy_attention))
            return original_attention(self,x,pe,attn_fn,prefix_len,transformer_options)
        dit.Attention.forward=projected
    if 'sdpa' in parts or 'heads' in parts:
        original_dispatch = dit.optimized_attention
        def plain(q,k,v,heads,mask,options,preferred):
            tensors=(dit.AttentionTensorContainer(t) for t in (q,k,v))
            return original_dispatch(*tensors,heads,mask=mask,transformer_options=options,preferred_attention=preferred)
        def by_head(q,k,v,heads,mask,options,preferred):
            outputs=[]
            for qi,ki,vi in zip(q.chunk(heads,-1),k.chunk(heads,-1),v.chunk(heads,-1)):
                outputs.append(plain(qi,ki,vi,1,mask,options,preferred))
            return torch.cat(outputs,-1)
        compiled_attention=torch.compile(by_head if 'heads' in parts else plain,backend='hpu_backend',dynamic=None,options={'force_static_compile':True})
        def dispatched(q,k,v,heads,mask=None,transformer_options={},preferred_attention=None,**kwargs):
            q,k,v=q.take(),k.take(),v.take()
            if q.device.type=='hpu' and q.dtype==torch.bfloat16 and not kwargs and 'optimized_attention_override' not in transformer_options:
                return compiled_attention(q,k,v,heads,mask,transformer_options,preferred_attention)
            tensors=(dit.AttentionTensorContainer(t) for t in (q,k,v))
            return original_dispatch(*tensors,heads,mask=mask,transformer_options=transformer_options,preferred_attention=preferred_attention,**kwargs)
        dit.optimized_attention=dispatched
