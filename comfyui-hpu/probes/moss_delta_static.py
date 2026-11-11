"""Isolated compiler-option comparison for the product recurrence."""
import torch
from comfy.text_encoders import hpu_gated_delta


def install():
    conv,_=hpu_gated_delta.compiled_ops()
    delta=torch.compile(hpu_gated_delta._decode_deferred,backend='hpu_backend',fullgraph=True,dynamic=False,
                        options={'force_static_compile':True})
    hpu_gated_delta.compiled_ops=lambda:(conv,delta)
