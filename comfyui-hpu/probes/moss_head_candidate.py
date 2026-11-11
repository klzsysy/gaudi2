"""Importable independent-head wrapper for Dynamo's module guards."""
import torch
import comfy.ops

original = comfy.ops.scaled_dot_product_attention


def head_one(q,k,v,*values,**kwargs):
    if q.ndim == 4 and q.shape[1] == 24 and q.shape[-1] == 128 and kwargs.get('attn_mask') is None:
        return torch.cat([original(q[:,h:h+1],k[:,h:h+1],v[:,h:h+1],*values,**kwargs)
                          for h in range(q.shape[1])],dim=1)
    return original(q,k,v,*values,**kwargs)


def install():
    comfy.ops.scaled_dot_product_attention = head_one
