"""Investigation-only long-sequence FP8 attention with fused probability quantization."""
import os
import torch


def fp8_attention(q, k, v):
    scales = [value.abs().amax().float().clamp_min(1e-12) / 240.0 for value in (q, k, v)]
    quantized = [torch.ops.hpu.cast_to_fp8_v2(value, scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
                 for value, scale in zip((q, k, v), scales)]
    probability_inverse = torch.full((), 1 / 128.0, device=q.device, dtype=torch.float32)
    outputs = []
    for head in range(q.shape[1]):
        query, key, value = [tensor[:, head:head+1] for tensor in quantized]
        scores = torch.ops.hpu.fp8_gemm_v2(query, False, key, True, None, q.dtype,
                                         scales[0] / q.shape[-1]**.5, scales[1], None, False)
        probabilities = torch.ops.hpu.softmax_fp8.Scalar_scales(scores, -1, 1.0, 128.0)
        outputs.append(torch.ops.hpu.fp8_gemm_v2(probabilities, False, value, False, None, q.dtype,
                                               probability_inverse, scales[2], None, False))
    return torch.cat(outputs, dim=1)


def heads(q, k, v, heads, comfy_attention, transformer_options):
    values = [value.reshape(value.shape[0], value.shape[1], heads, -1).transpose(1, 2) for value in (q, k, v)]
    output = fp8_attention(*values)
    return output.transpose(1, 2).flatten(2)


def variant_heads(q, k, v, heads, comfy_attention, transformer_options):
    values = [value.reshape(value.shape[0], value.shape[1], heads, -1).transpose(1, 2) for value in (q, k, v)]
    output = fp8_attention_variant(*values, os.environ.get('MOSS_DEEP_ATTENTION_MODE', 'head'))
    return output.transpose(1, 2).flatten(2)


def fp8_attention_variant(q, k, v, mode):
    outputs = []
    for head in range(q.shape[1]):
        query, key, value = [tensor[0, head] for tensor in (q, k, v)]
        if 'center' in mode:
            key = (key.float() - key.float().mean(dim=0, keepdim=True)).to(key.dtype)
        reduction = -1 if 'row' in mode else (-2, -1)
        query_scale = query.abs().amax(reduction, keepdim=True).float().clamp_min(1e-12) / 240.0
        key_scale = key.abs().amax(reduction, keepdim=True).float().clamp_min(1e-12) / 240.0
        value_scale = value.abs().amax().float().clamp_min(1e-12) / 240.0
        value_quantized = torch.ops.hpu.cast_to_fp8_v2(value, value_scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
        if mode.startswith('av'):
            scores = (query / q.shape[-1]**.25) @ (key / q.shape[-1]**.25).transpose(-1, -2)
        else:
            query_quantized = torch.ops.hpu.cast_to_fp8_v2(query, query_scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
            key_quantized = torch.ops.hpu.cast_to_fp8_v2(key, key_scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
            scores = torch.ops.hpu.fp8_gemm_v2(query_quantized, False, key_quantized, True, None, q.dtype,
                                             query_scale / q.shape[-1]**.5, key_scale.transpose(-1, -2), None, False)
        denominator = None
        if 'exp' in mode:
            work = scores.float() if '32' in mode else scores
            exponentials = (work - work.amax(-1, keepdim=True)).exp()
            denominator = exponentials.sum(-1, keepdim=True, dtype=torch.float32)
            probabilities = torch.ops.hpu.cast_to_fp8_v2(exponentials, 128.0, False, False, torch.float8_e4m3fn)[0]
        elif 'split' in mode:
            probabilities = torch.ops.hpu.cast_to_fp8_v2(scores.softmax(-1), 128.0, False, False, torch.float8_e4m3fn)[0]
        else:
            probabilities = torch.ops.hpu.softmax_fp8.Scalar_scales(scores, -1, 1.0, 128.0)
        probability_inverse = torch.full((), 1 / 128.0, device=q.device, dtype=torch.float32)
        output = torch.ops.hpu.fp8_gemm_v2(probabilities, False, value_quantized, False, None, q.dtype,
                                         probability_inverse, value_scale, None, False)
        if denominator is not None:
            output = (output.float() / denominator).to(q.dtype)
        outputs.append(output.unsqueeze(0).unsqueeze(0))
    return torch.cat(outputs, dim=1)


def install():
    from comfy.ldm.moss_image import model as moss
    moss._attention_heads = heads if os.environ.get('MOSS_DEEP_ATTENTION_MODE', 'global') == 'global' else variant_heads
