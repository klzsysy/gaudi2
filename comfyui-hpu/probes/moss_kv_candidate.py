"""Investigation-only fixed KV attention candidates; install in an isolated process."""
from functools import lru_cache
import os

import torch
import torch.nn.functional as F
import comfy_kitchen


def plain(q, key, value, length):
    return comfy_kitchen.flash_attention_decode_gqa(q, key, value, length)


def tree(q, key, value, length, k, v, mask):
    out, lse = comfy_kitchen.flash_attention_decode_gqa(q, key, value, length, return_lse=True, causal=False)
    return comfy_kitchen.flash_attention_decode_tree_merge(out, lse, q, k, v, mask, torch.empty_like(out))


def sdpa(q, key, value, length, mask=None):
    rows = q.shape[2]
    relative = torch.arange(key.shape[2], device=q.device) - length.view(-1, 1, 1, 1)
    if mask is None:
        visible = relative < torch.arange(1 - rows, 1, device=q.device).view(rows, 1)
    else:
        visible = (relative < 0) | ((relative < rows) & ((mask.view(rows, 1) >> relative.clamp(0, rows - 1)) & 1).bool())
    bias = torch.zeros(visible.shape, dtype=q.dtype, device=q.device).masked_fill_(~visible, float('-inf'))
    out = F.scaled_dot_product_attention(q, key, value, attn_mask=bias, enable_gqa=q.shape[1] != key.shape[1])
    return out.transpose(1, 2).reshape(q.shape[0], rows, -1)


@lru_cache(None)
def compiled(function):
    return torch.compile(function, backend='hpu_backend', dynamic=None, options={'force_static_compile': True})


def attend(cache, q, k, v, mode):
    topology = cache.tracker['tree']
    key, value = cache.key, cache.value
    if 'prefix' in mode:
        bucket = int(os.environ.get('MOSS_KV_BUCKET', '256'))
        limit = min(key.shape[2], ((cache.index + q.shape[2] + bucket - 1) // bucket) * bucket)
        key, value = key[:, :, :limit], value[:, :, :limit]
    if mode.startswith('sdpa'):
        function = compiled(sdpa) if 'compile' in mode else sdpa
        return function(q, key, value, cache.seqlen, None if topology is None else topology.mask)
    function = plain if topology is None else tree
    if mode.startswith('compile'):
        function = compiled(function)
    if topology is None:
        return function(q, key, value, cache.seqlen)
    return function(q, key, value, cache.seqlen, k, v, topology.mask)


def install(mode):
    from comfy.text_encoders.llama import FixedKVCache
    original = FixedKVCache.attend
    if mode != 'baseline':
        FixedKVCache.attend = lambda cache, q, k, v: attend(cache, q, k, v, mode)
    return original
