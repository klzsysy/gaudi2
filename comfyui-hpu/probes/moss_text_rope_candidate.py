"""Expose the existing kitchen RoPE implementation to HPU AOT compilation."""
import torch
import comfy_kitchen
from comfy_kitchen.backends.eager.rope import apply_rope_split_half as eager_rope
from habana_frameworks.torch.dynamo.compile_backend.decomposition import hpu_backend_decompositions_common


def rope_one(x, frequencies, foreach=False):
    # Functional form of kitchen's eager split-half RoPE, preserving its two products.
    value = x if torch.promote_types(x.dtype, frequencies.dtype) == frequencies.dtype else x.to(frequencies.dtype)
    if frequencies.ndim == 2:
        frequencies = frequencies.unsqueeze(0)
    diagonal = frequencies.diagonal(dim1=-2, dim2=-1).movedim(-1, -2)
    products = value.unflatten(-1, (2, -1)) * diagonal
    half = x.shape[-1] // 2
    if foreach:
        first, second = torch._foreach_addcmul(
            [products[..., 0, :], products[..., 1, :]],
            [value[..., half:].to(products.dtype), value[..., :half].to(products.dtype)],
            [frequencies[..., 0, 1], frequencies[..., 1, 0]])
    else:
        first = torch.addcmul(products[..., 0, :], value[..., half:], frequencies[..., 0, 1])
        second = torch.addcmul(products[..., 1, :], value[..., :half], frequencies[..., 1, 0])
    return torch.cat((first, second), dim=-1).to(x.dtype)


def apply_rope_split_half(query, key, frequencies):
    return rope_one(query, frequencies), rope_one(key, frequencies)


def apply_rope_split_half_foreach(query, key, frequencies):
    return rope_one(query, frequencies, True), rope_one(key, frequencies, True)


def install(mode='functional'):
    if mode in ('opaque-none', 'opaque-functional-none'):
        hpu_backend_decompositions_common.pop(torch.ops.comfy_kitchen.apply_rope_split_half.default, None)
        function = eager_rope if mode == 'opaque-none' else apply_rope_split_half
        settings = {'force_static_compile': True, 'decomposition_mode': 'none'}
        compiled = torch.compile(function, backend='hpu_backend', dynamic=None, options=settings)
        torch.library.register_kernel('comfy_kitchen::apply_rope_split_half', 'hpu', compiled)
        return
    function = apply_rope_split_half_foreach if mode == 'foreach' else apply_rope_split_half
    hpu_backend_decompositions_common[torch.ops.comfy_kitchen.apply_rope_split_half.default] = function
