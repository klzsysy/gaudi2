"""Investigation-only larger decode regions around existing Qwen operations."""
import functools
import os

import torch
from comfy.text_encoders.qwen35 import GatedAttention, GatedDeltaNet


def install():
    mode = os.environ.get('MOSS_DECODE_REGIONS', '')
    for name, cls in (('attention', GatedAttention), ('delta', GatedDeltaNet)):
        if name not in mode and mode != 'both':
            continue
        original = cls.forward
        compiled = torch.compile(original, backend='hpu_backend', dynamic=None, options={'force_static_compile': True})
        def make_forward(original, compiled, name):
            @functools.wraps(original)
            def forward(module, x, past_key_value=None, **kwargs):
                eligible = past_key_value is not None and x.shape[1] <= 8
                if name == 'delta':
                    eligible = eligible and past_key_value.decode_step(x.shape[1])
                else:
                    eligible = eligible and kwargs.get('attention_mask') is None
                function = compiled if eligible else original
                return function(module, x, past_key_value=past_key_value, **kwargs)
            return forward
        cls.forward = make_forward(original, compiled, name)
