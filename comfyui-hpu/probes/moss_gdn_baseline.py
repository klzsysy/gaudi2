"""Restore the pre-parallel public Kitchen decode for isolated A/B processes."""
from functools import lru_cache
import torch
import comfy_kitchen
from comfy.text_encoders import hpu_gated_delta,qwen35

product_ops = hpu_gated_delta.compiled_ops
product_region = qwen35._compiled_hpu_delta


def disable_region():
    qwen35._compiled_hpu_delta = lambda:qwen35.GatedDeltaNet._forward


@lru_cache(maxsize=1)
def baseline_ops():
    conv,_ = product_ops()
    delta = torch.compile(comfy_kitchen.gated_delta_decode_deferred,backend='hpu_backend',fullgraph=True,dynamic=False)
    return conv,delta


def install():
    hpu_gated_delta.compiled_ops = baseline_ops
    disable_region()
