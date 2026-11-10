"""Investigation-only compiled projections, preserving modulation and residuals."""
from functools import lru_cache

import torch
import comfy.ops
from comfy.ldm.moss_image import model as moss


def attention_projection(layer, y, pe, transformer_options, ref_kv=None, kv_out=None, mask=None):
    if layer.fused_qkv:
        q, k, v = layer.proj['qkv'](y).chunk(3, dim=-1)
    else:
        q, k, v = layer.proj['q'](y), layer.proj['k'](y), layer.proj['v'](y)
    out = moss._attention(q, k, v, layer.q_norm, layer.k_norm, pe, layer.num_heads, layer.comfy_attention, transformer_options, ref_kv, kv_out, mask)
    return layer.proj['o'](out)


def feed_forward(layer, y):
    y = layer.proj['ffn_up_gate'](y)
    return comfy.ops.linear_input_act(layer.proj['ffn_down'], y, 'situ')


@lru_cache(maxsize=2)
def compiled_regions(dynamic=None):
    return (torch.compile(attention_projection, backend='hpu_backend', dynamic=dynamic),
            torch.compile(feed_forward, backend='hpu_backend', dynamic=dynamic))


def forward(layer, x, mod, pe, transformer_options={}, ref=None, ref_kv=None, kv_out=None, mask=None):
    project, ffn = compiled_regions()
    y = moss.modulate(x, layer.norm1, mod, 0, ref)
    x = moss.residual(x, project(layer, y, pe, transformer_options, ref_kv, kv_out, mask), mod, 2, ref)
    y = moss.modulate(x, layer.norm2, mod, 3, ref)
    return moss.residual(x, ffn(layer, y), mod, 5, ref)


def fp16_wanted(x):
    return x.is_cuda and x.dtype == torch.float16 and getattr(torch.backends.cuda.matmul, 'allow_fp16_accumulation', False)
