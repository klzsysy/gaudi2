"""Process-local regional compilation experiment for Qwen Image 2.1."""
import torch
import comfy.model_management
import comfy.ops
from comfy.ldm.qwen_image21 import model


def install():
    for cls in (model.Attention, model.SwiGLUFeedForward):
        original = cls.forward
        compiled = torch.compile(original, backend='hpu_backend', dynamic=None, options={'force_static_compile':True})
        def forward(self, x, *args, _original=original, _compiled=compiled, **kwargs):
            if x.device.type == 'hpu' and x.dtype == torch.bfloat16 and not comfy.model_management.args.disable_comfy_compiler and not comfy.model_management.in_training and not torch.is_grad_enabled() and not comfy.ops.has_forward_hooks(self):
                return _compiled(self, x, *args, **kwargs)
            return _original(self, x, *args, **kwargs)
        cls.forward = forward
