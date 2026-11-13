"""Compare product and pre-change DiT operations on the same live block inputs."""
import json
import os
from pathlib import Path
import torch
import comfy.ops
from comfy.ldm.qwen_image21 import model as dit


def install():
    original = dit.QwenImage21TransformerBlock.forward
    native = dit._hpu_rms_rope
    compiled_attention = dit._compiled_attention
    counts = {}
    output = Path(os.environ['QWEN_DEEP_DIR'])/'block-quality.jsonl'
    def checked(self,x,mod,pe,attn_fn,prefix_len,transformer_options={}):
        layer = transformer_options.get('block_index',-1)
        if layer not in (0,16,31):
            return original(self,x,mod,pe,attn_fn,prefix_len,transformer_options)
        target = x.shape[1] >= 1000
        step = counts.get(layer,0)
        if target:
            counts[layer]=step+1
        if target and step not in (0,4,7):
            return original(self,x,mod,pe,attn_fn,prefix_len,transformer_options)
        try:
            dit._hpu_rms_rope = comfy.ops.quant_ops.ck.rms_rope
            dit._compiled_attention = lambda: dit._attention
            expected=original(self,x.clone(),mod,pe,attn_fn,prefix_len,transformer_options).cpu().float()
        finally:
            dit._hpu_rms_rope=native
            dit._compiled_attention=compiled_attention
        result=original(self,x,mod,pe,attn_fn,prefix_len,transformer_options)
        actual=result.cpu().float()
        error=actual-expected
        record=dict(layer=layer,step=step,target=target,shape=list(x.shape),
                    exact=bool(torch.equal(actual,expected)),finite=bool(torch.isfinite(actual).all()),
                    max_abs=float(error.abs().max()),rrmse=float(error.square().mean().sqrt()/expected.square().mean().sqrt()))
        with output.open('a') as stream:stream.write(json.dumps(record)+'\n')
        return result
    dit.QwenImage21TransformerBlock.forward=checked
