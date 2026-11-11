"""Isolate symbolic-shape promotion in the short Qwen compilation regions."""
import torch
from comfy.text_encoders import qwen35,hpu_mlp


def install(scope='both'):
    if scope in ('both','attention'):
        attention=torch.compile(qwen35._hpu_attention,backend='hpu_backend',dynamic=False,options={'force_static_compile':True})
        qwen35._compiled_hpu_attention=lambda:attention
    if scope in ('both','mlp'):
        mlp=torch.compile(hpu_mlp._forward,backend='hpu_backend',dynamic=False,options={'force_static_compile':True})
        hpu_mlp._compiled_forward=lambda:mlp
