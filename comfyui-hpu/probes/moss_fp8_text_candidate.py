"""Investigation-only native FP8 Qwen operations and existing region activation."""
import os
import torch
import comfy.ops
from comfy import sd1_clip
from comfy.text_encoders import qwen35

def install():
    if os.environ.get('MOSS_FP8_LINEAR_COMPILE') == '1':
        comfy.ops.quant_ops.hpu_fp8_linear = torch.compile(comfy.ops.quant_ops.hpu_fp8_linear, backend='hpu_backend', dynamic=True, options={'force_static_compile': True})
    original_clip_init = sd1_clip.SDClipModel.__init__
    def clip_init(self, *args, **kwargs):
        options = dict(kwargs.get('model_options', {}))
        quantization = options.get('quantization_metadata')
        if quantization and options.get('custom_operations') is None:
            options['custom_operations'] = comfy.ops.mixed_precision_ops(
                quantization, kwargs.get('dtype'), load_device=torch.device('hpu'))
            kwargs['model_options'] = options
        original_clip_init(self, *args, **kwargs)
        if os.environ.get('MOSS_FP8_TEXT_REGIONS', '1') == '1':
            for module in self.transformer.modules():
                if isinstance(module, (qwen35.GatedAttention, qwen35.GatedDeltaNet)):
                    module.hpu_compile_decode = True
                if isinstance(module, qwen35.Qwen35TransformerBlock):
                    module.hpu_compile_mlp = True
    sd1_clip.SDClipModel.__init__ = clip_init
