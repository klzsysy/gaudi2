"""Investigation-only MOSS FP8 region and RoPE candidates."""
import os
import functools

import torch
from comfy.ldm.moss_image import model as moss
from comfy.ldm.modules import attention as attention_module


original_forward = moss.Block.forward
original_rope = moss.hpu._rope
original_rms_rope = moss.hpu.rms_rope_split_half
original_av = attention_module._hpu_fp8_attention_av
original_heads = moss._attention_heads
original_situ = moss.comfy.ops.INPUT_ACT_EAGER['situ']


def baseline_norm(q, k, pe, q_scale, k_scale, eps, blockwise=False):
    q = torch.nn.functional.rms_norm(q, (q.shape[-1],), q_scale, eps)
    k = torch.nn.functional.rms_norm(k, (k.shape[-1],), k_scale, eps)
    return original_rope(q, pe, blockwise), original_rope(k, pe, blockwise)


def baseline_block(self, x, mod, pe, transformer_options={}, ref=None, ref_kv=None, kv_out=None, mask=None):
    attention, ffn = moss._compiled_block_ops()
    out = attention(self, moss.modulate(x, self.norm1, mod, 0, ref), pe, transformer_options, ref_kv, kv_out, mask)
    x = moss.residual(x, out, mod, 2, ref)
    del out
    return moss.residual(x, ffn(self, moss.modulate(x, self.norm2, mod, 3, ref)), mod, 5, ref)


def single_head_av(q, k, v):
    query, key, value = q[0, 0], k[0, 0], v[0, 0]
    value_scale = value.abs().amax().float().clamp_min(1e-12) / 240.0
    quantized = torch.ops.hpu.cast_to_fp8_v2(value, value_scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
    scale = q.shape[-1] ** -.25
    scores = (query * scale) @ (key * scale).transpose(-1, -2)
    p = torch.ops.hpu.softmax_fp8.Scalar_scales(scores, -1, 1.0, 128.0)
    pscale = torch.full((), 1 / 128.0, device=q.device, dtype=torch.float32)
    out = torch.ops.hpu.fp8_gemm_v2(p, False, quantized, False, None, q.dtype, pscale, value_scale, None, False)
    return out.unsqueeze(0).unsqueeze(0)


def single_head_dispatch(q, k, v, heads, comfy_attention, transformer_options):
    outputs = []
    for query, key, value in zip(q.chunk(heads, dim=-1), k.chunk(heads, dim=-1), v.chunk(heads, dim=-1)):
        tensors = (moss.AttentionTensorContainer(tensor) for tensor in (query, key, value))
        outputs.append(moss.optimized_attention(*tensors, 1, transformer_options=transformer_options,
                                               preferred_attention=comfy_attention))
    return torch.cat(outputs, dim=-1)


def situ_shared_tanh(x):
    hidden, gate = x.float().chunk(2, dim=-1)
    t = torch.tanh(gate * .25)
    gated = 2 * t * (1 + t).square() / (1 + t.square())
    return (gated * (25 * torch.tanh(hidden / 25))).to(x.dtype)


def value_cast_attention(q, k, v, mode):
    scales = v.abs().amax((-2, -1), keepdim=True).float().clamp_min(1e-12) / 240
    if mode == 'value-flat':
        row_scales = scales.expand(*v.shape[:-1], 1).reshape(-1, 1)
        quantized = torch.ops.hpu.cast_to_fp8_v2(v.reshape(-1, v.shape[-1]), row_scales.reciprocal(),
                                               False, False, torch.float8_e4m3fn)[0].reshape(v.shape)
    else:
        normalized = v * scales.reciprocal()
        quantized = torch.ops.hpu.cast_to_fp8_v2(normalized, 1.0, False, False, torch.float8_e4m3fn)[0]
    outputs = []
    for head in range(q.shape[1]):
        query, key = q[0, head], k[0, head]
        scale = q.shape[-1] ** -.25
        scores = (query * scale) @ (key * scale).transpose(-1, -2)
        p = torch.ops.hpu.softmax_fp8.Scalar_scales(scores, -1, 1.0, 128.0)
        pscale = torch.full((), 1 / 128.0, device=q.device, dtype=torch.float32)
        out = torch.ops.hpu.fp8_gemm_v2(p, False, quantized[0, head], False, None, q.dtype,
                                      pscale, scales[0, head], None, False)
        outputs.append(out.unsqueeze(0).unsqueeze(0))
    return torch.cat(outputs, dim=1)


def shared_amax_attention(q, k, v):
    scales = v.abs().amax((-2, -1)).float().clamp_min(1e-12) / 240
    outputs = []
    for head in range(q.shape[1]):
        query, key, value = q[0, head], k[0, head], v[0, head]
        value_scale = scales[0, head]
        quantized = torch.ops.hpu.cast_to_fp8_v2(value, value_scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
        scale = q.shape[-1] ** -.25
        scores = (query * scale) @ (key * scale).transpose(-1, -2)
        p = torch.ops.hpu.softmax_fp8.Scalar_scales(scores, -1, 1.0, 128.0)
        pscale = torch.full((), 1 / 128.0, device=q.device, dtype=torch.float32)
        out = torch.ops.hpu.fp8_gemm_v2(p, False, quantized, False, None, q.dtype,
                                      pscale, value_scale, None, False)
        outputs.append(out)
    return torch.stack(outputs, dim=1).unsqueeze(0).transpose(1, 2)


def shared_value_heads(q, k, v, heads, comfy_attention, transformer_options):
    queries, keys, values = [value.reshape(value.shape[0], value.shape[1], heads, -1).transpose(1, 2)
                             for value in (q, k, v)]
    value_scales = values.abs().amax((-2, -1), keepdim=True).float().clamp_min(1e-12) / 240.0
    quantized_values = torch.ops.hpu.cast_to_fp8_v2(values, value_scales.reciprocal(), False, False, torch.float8_e4m3fn)[0]
    outputs = []
    for head in range(heads):
        query, key, value = queries[0, head], keys[0, head], quantized_values[0, head]
        scale = query.shape[-1] ** -.25
        scores = (query * scale) @ (key * scale).transpose(-1, -2)
        probabilities = torch.ops.hpu.softmax_fp8.Scalar_scales(scores, -1, 1.0, 128.0)
        probability_scale = torch.full((), 1 / 128.0, device=q.device, dtype=torch.float32)
        output = torch.ops.hpu.fp8_gemm_v2(probabilities, False, value, False, None, q.dtype,
                                        probability_scale, value_scales[0, head], None, False)
        outputs.append(output)
    return torch.cat(outputs, dim=-1).unsqueeze(0)


def scaled_softmax_av(q, k, v):
    query, key, value = q[0, 0], k[0, 0], v[0, 0]
    value_scale = value.abs().amax().float().clamp_min(1e-12) / 240.0
    quantized = torch.ops.hpu.cast_to_fp8_v2(value, value_scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
    scores = query @ key.transpose(-1, -2)
    inverse_heads = torch.full((), q.shape[-1]**-.5, device=q.device, dtype=torch.float32)
    probabilities = torch.ops.hpu.softmax_fp8.Scalar_scales(scores, -1, 1.0, 128.0, inverse_heads)
    probability_scale = torch.full((), 1 / 128.0, device=q.device, dtype=torch.float32)
    output = torch.ops.hpu.fp8_gemm_v2(probabilities, False, quantized, False, None, q.dtype,
                                    probability_scale, value_scale, None, False)
    return output.unsqueeze(0).unsqueeze(0)


def attention_region(self, x, mod, pe, options):
    return self._forward_attention(moss.modulate(x, self.norm1, mod, 0, None), pe, options)


def ffn_region(self, x, mod):
    return self._forward_ffn(moss.modulate(x, self.norm2, mod, 3, None))


def ffn_residual_region(self, x, mod):
    return moss.residual(x, ffn_region(self, x, mod), mod, 5, None)


def whole(self, x, mod, pe, options):
    out = attention_region(self, x, mod, pe, options)
    x = moss.residual(x, out, mod, 2, None)
    return moss.residual(x, ffn_region(self, x, mod), mod, 5, None)


def native_rope(x, pe, blockwise=False):
    if not blockwise:
        return original_rope(x, pe, blockwise)
    cos = torch.cat((pe[..., 0, 0], pe[..., 1, 1]), dim=-1)
    sin = torch.cat((-pe[..., 0, 1], pe[..., 1, 0]), dim=-1)
    return torch.ops.hpu.rotary_pos_embedding(x, sin, cos, None, 0, 0)


def native_norm(q, k, pe, q_scale, k_scale, eps, blockwise=False):
    q = torch.ops.hpu.rms_norm(q, q_scale, eps)[0]
    k = torch.ops.hpu.rms_norm(k, k_scale, eps)[0]
    return moss.hpu._rope(q, pe, blockwise), moss.hpu._rope(k, pe, blockwise)


def native_norm_separate(q, k, pe, q_scale, k_scale, eps, blockwise=False):
    weight = torch.ones(q.shape[-1], dtype=q.dtype, device=q.device)
    q = torch.ops.hpu.rms_norm(q, weight, eps)[0] * q_scale
    k = torch.ops.hpu.rms_norm(k, weight, eps)[0] * k_scale
    return moss.hpu._rope(q, pe, blockwise), moss.hpu._rope(k, pe, blockwise)


def native_norm_fp32(q, k, pe, q_scale, k_scale, eps, blockwise=False):
    q = torch.ops.hpu.rms_norm(q.float(), q_scale.float(), eps)[0].to(q.dtype)
    k = torch.ops.hpu.rms_norm(k.float(), k_scale.float(), eps)[0].to(k.dtype)
    return moss.hpu._rope(q, pe, blockwise), moss.hpu._rope(k, pe, blockwise)


def native_norm_unrounded(q, k, pe, q_scale, k_scale, eps, blockwise=False):
    dtype = q.dtype
    q = torch.ops.hpu.rms_norm(q.float(), q_scale.float(), eps)[0]
    k = torch.ops.hpu.rms_norm(k.float(), k_scale.float(), eps)[0]
    return moss.hpu._rope(q, pe, blockwise).to(dtype), moss.hpu._rope(k, pe, blockwise).to(dtype)


def install(mode=None):
    mode = mode or os.environ.get('MOSS_FUSION_MODE', 'whole')
    moss.Block.forward = original_forward
    moss.hpu._rope = native_rope if 'rope' in mode else original_rope
    moss.hpu.rms_rope_split_half = native_norm if 'norm' in mode else original_rms_rope
    attention_module._hpu_fp8_attention_av = scaled_softmax_av if 'scaled' in mode else original_av
    moss._attention_heads = shared_value_heads if 'shared-v' in mode else original_heads
    moss.comfy.ops.INPUT_ACT_EAGER['situ'] = situ_shared_tanh if 'situ' in mode else original_situ
    if mode == 'baseline-f67':
        moss.Block.forward = baseline_block
        moss.hpu.rms_rope_split_half = baseline_norm
        moss._attention_heads = single_head_dispatch
        attention_module._hpu_fp8_attention_av = single_head_av
        return
    if mode.startswith('value-'):
        attention_module._hpu_fp8_attention_av = functools.partial(value_cast_attention, mode=mode)
    elif mode == 'shared-amax':
        attention_module._hpu_fp8_attention_av = shared_amax_attention
    if 'unshared' in mode:
        attention_module._hpu_fp8_attention_av = single_head_av
        moss._attention_heads = single_head_dispatch
    if 'norm-separate' in mode:
        moss.hpu.rms_rope_split_half = native_norm_separate
    elif 'norm-fp32' in mode:
        moss.hpu.rms_rope_split_half = native_norm_fp32
    elif 'norm-unrounded' in mode:
        moss.hpu.rms_rope_split_half = native_norm_unrounded
    if mode.startswith('whole'):
        compiled = torch.compile(whole, backend='hpu_backend', dynamic=None, options={'force_static_compile': True})
        def forward(self, x, mod, pe, transformer_options={}, ref=None, ref_kv=None, kv_out=None, mask=None):
            if ref is not None or ref_kv is not None or kv_out is not None or mask is not None:
                return original_forward(self, x, mod, pe, transformer_options, ref, ref_kv, kv_out, mask)
            return compiled(self, x, mod, pe, transformer_options)
        moss.Block.forward = forward
    elif mode.startswith('ffn'):
        fused_residual = 'residual' in mode
        ffn = torch.compile(ffn_residual_region if fused_residual else ffn_region, backend='hpu_backend',
                            dynamic=None, options={'force_static_compile': True})
        def forward(self, x, mod, pe, transformer_options={}, ref=None, ref_kv=None, kv_out=None, mask=None):
            if ref is not None or ref_kv is not None or kv_out is not None or mask is not None:
                return original_forward(self, x, mod, pe, transformer_options, ref, ref_kv, kv_out, mask)
            attention = moss._compiled_block_ops()[0]
            out = attention(self, moss.modulate(x, self.norm1, mod, 0, None), pe, transformer_options)
            x = moss.residual(x, out, mod, 2, None)
            if fused_residual:
                return ffn(self, x, mod)
            return moss.residual(x, ffn(self, x, mod), mod, 5, None)
        moss.Block.forward = forward
    elif mode == 'regions':
        attention = torch.compile(attention_region, backend='hpu_backend', dynamic=None, options={'force_static_compile': True})
        ffn = torch.compile(ffn_region, backend='hpu_backend', dynamic=None, options={'force_static_compile': True})
        def forward(self, x, mod, pe, transformer_options={}, ref=None, ref_kv=None, kv_out=None, mask=None):
            if ref is not None or ref_kv is not None or kv_out is not None or mask is not None:
                return original_forward(self, x, mod, pe, transformer_options, ref, ref_kv, kv_out, mask)
            x = moss.residual(x, attention(self, x, mod, pe, transformer_options), mod, 2, None)
            return moss.residual(x, ffn(self, x, mod), mod, 5, None)
        moss.Block.forward = forward

