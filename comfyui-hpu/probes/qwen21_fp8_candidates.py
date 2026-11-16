"""Process-local FP8 execution candidates, preserving workflow parameters."""
import functools
import os
from pathlib import Path
import torch
from comfy.text_encoders import qwen35
from comfy.ldm.qwen_image21 import model as dit


def install(mode):
    parts = mode.split('+')
    if 'decode' in parts or 'regions' in parts:
        original_gdn = qwen35.GatedDeltaNet.__init__
        original_block = qwen35.Qwen35TransformerBlock.__init__

        @functools.wraps(original_gdn)
        def gdn(self, config, device=None, dtype=None, ops=None):
            original_gdn(self, config, device=device, dtype=dtype, ops=ops)
            self.hpu_compile_decode_core |= getattr(ops, 'hpu_fp8', False)

        @functools.wraps(original_block)
        def block(self, config, index, device=None, dtype=None, ops=None):
            original_block(self, config, index, device=device, dtype=dtype, ops=ops)
            self.hpu_compile_mlp_residual |= getattr(ops, 'hpu_fp8', False)

        qwen35.GatedDeltaNet.__init__ = gdn
        qwen35.Qwen35TransformerBlock.__init__ = block
        if 'regions' in parts:
            qwen35.Qwen35._generation_forward = lambda self, embeds, caches: self.model.forward
    if 'ffn' in parts:
        original = dit.SwiGLUFeedForward.__init__

        @functools.wraps(original)
        def ffn(self, *args, **kwargs):
            original(self, *args, **kwargs)
            self.hpu_compile = True

        dit.SwiGLUFeedForward.__init__ = ffn

    if 'capture_attention' in parts:
        original = dit.attention
        captured = False

        def capture(q, k, v, heads, mask=None, transformer_options={}, preferred_attention=None):
            nonlocal captured
            if not captured and mask is None and q.peek().shape[1] >= 4096:
                torch.save(dict(q=q.peek().cpu(), k=k.peek().cpu(), v=v.peek().cpu(), heads=heads),
                           Path(os.environ['QWEN_DEEP_DIR']) / 'attention.pt')
                captured = True
            return original(q, k, v, heads, mask, transformer_options, preferred_attention)

        dit.attention = capture

    if 'capture_ffn' in parts:
        original_ffn = dit.SwiGLUFeedForward.forward
        ffn_index = 0

        def capture_ffn(self, x):
            nonlocal ffn_index
            if x.shape[1] >= 4096:
                ffn_index += 1
                if ffn_index in (1,17,32):
                    torch.save(dict(state={k:v.cpu() for k,v in self.state_dict().items()},x=x.cpu()),
                               Path(os.environ['QWEN_DEEP_DIR']) / f'ffn-{ffn_index}.pt')
            return original_ffn(self,x)

        dit.SwiGLUFeedForward.forward = capture_ffn

    if 'fast_attention' in parts:
        from habana_frameworks.torch.hpex.kernels import FusedSDPA
        original_attention = dit._attention

        def attention(q,k,v,heads,mask,transformer_options,preferred_attention):
            if mask is not None:
                return original_attention(q,k,v,heads,mask,transformer_options,preferred_attention)
            batch,length,_ = q.shape
            q,k,v = [x.view(batch,x.shape[1],heads,-1).transpose(1,2) for x in (q,k,v)]
            out = FusedSDPA.apply(q,k,v,None,0.0,False,None,'fast',False)
            return out.transpose(1,2).reshape(batch,length,-1)

        dit._attention = attention
