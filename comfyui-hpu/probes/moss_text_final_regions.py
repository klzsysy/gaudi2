"""Last launch-bound text candidates around unchanged existing operations."""
import functools
import torch
from comfy.text_encoders import qwen35
from comfy.text_encoders.llama import BaseGenerate


def native_norm(module, x):
    return torch.ops.hpu.rms_norm(x, module.scale().to(x), module.eps)[0]


def install(mode):
    if mode == 'native_norm':
        qwen35.RMSNorm.forward = torch.compile(native_norm, backend='hpu_backend', dynamic=None, options={'force_static_compile': True})
        return
    if mode in ('outer_norm', '3d_norm'):
        original = qwen35.RMSNorm.forward
        def norm(module, x):
            if torch.compiler.is_compiling() or mode == '3d_norm' and x.ndim != 3:
                return qwen35.LlamaRMSNorm.forward(module, x)
            return original(module, x)
        qwen35.RMSNorm.forward = norm
        return
    if mode == 'probability':
        BaseGenerate.processed_probs = torch.compile(BaseGenerate.processed_probs, backend='hpu_backend', dynamic=None, options={'force_static_compile': True})
        return
    cls = qwen35.Qwen35TransformerBlock if mode == 'block' else qwen35.MTPHead
    original = cls.forward
    compiled = torch.compile(original, backend='hpu_backend', dynamic=None, options={'force_static_compile': True})
    @functools.wraps(original)
    def forward(module, x, *values, **kwargs):
        eligible = x.shape[1] <= 8
        if mode == 'block':
            cache = kwargs.get('past_key_value')
            eligible = eligible and cache is not None and cache.index > 0
        return (compiled if eligible else original)(module,x,*values,**kwargs)
    cls.forward = forward
