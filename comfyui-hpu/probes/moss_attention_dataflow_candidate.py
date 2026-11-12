"""Process-local mixed-attention graph experiments with unchanged FP8 scales."""
import torch
from comfy.ldm.modules import attention


original_attention = attention._hpu_fp8_attention_av


def candidate(mode):
    query_tile = int(mode.rsplit('-', 1)[1]) if mode.startswith('batch-query-') else None

    def forward(q, k, v):
        if query_tile is not None and 'linear' not in mode:
            torch._dynamo.mark_static(q, 2)
        query, key, value = q[0, 0], k[0, 0], v[0, 0]
        value_scale = value.abs().amax().float().clamp_min(1e-12) / 240.0
        quantized_value = torch.ops.hpu.cast_to_fp8_v2(
            value, value_scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
        scale = q.shape[-1] ** -0.25
        query, key = query * scale, key * scale
        if query_tile is not None:
            length = query.shape[0]
            padding = ((length + query_tile - 1) // query_tile) * query_tile - length if 'linear' in mode else -length % query_tile
            query = torch.nn.functional.pad(query, (0, 0, 0, padding)).reshape(-1, query_tile, query.shape[-1])
            batches = query.shape[0]
            scores = torch.bmm(query, key.transpose(-1, -2).unsqueeze(0).expand(batches, -1, -1))
            if 'softmax2d' in mode:
                scores = scores.flatten(0, 1)
            probabilities = torch.ops.hpu.softmax_fp8.Scalar_scales(scores, -1, 1.0, 128.0)
            probability_scale = torch.full((), 1 / 128.0, device=q.device, dtype=torch.float32)
            if 'av2d' in mode:
                probabilities = probabilities.reshape(-1, length)
                output = torch.ops.hpu.fp8_gemm_v2(
                    probabilities, False, quantized_value, False, None, q.dtype,
                    probability_scale, value_scale, None, False).reshape(batches, query_tile, -1)
            else:
                probabilities = probabilities.reshape(batches, query_tile, length)
                output = torch.ops.hpu.fp8_gemm_v2(
                    probabilities, False, quantized_value.unsqueeze(0).expand(batches, -1, -1), False,
                    None, q.dtype, probability_scale, value_scale, None, False)
            return output.flatten(0, 1)[:length].unsqueeze(0).unsqueeze(0)
        swapped_scores = mode in ('swap-qk', 'swap-both')
        if swapped_scores:
            scores = key @ query.transpose(-1, -2)
        elif mode == 'rank3':
            scores = torch.bmm(query.unsqueeze(0), key.transpose(-1, -2).unsqueeze(0))
        else:
            scores = query @ key.transpose(-1, -2)
        probabilities = torch.ops.hpu.softmax_fp8.Scalar_scales(
            scores, -2 if swapped_scores else -1, 1.0, 128.0)
        if mode == 'rank3':
            probabilities = probabilities.squeeze(0)
        probability_scale = torch.full((), 1 / 128.0, device=q.device, dtype=torch.float32)
        if mode == 'combined-scale':
            value_scale = value_scale / 128.0
            probability_scale = None
        if mode in ('swap-av', 'swap-both'):
            output = torch.ops.hpu.fp8_gemm_v2(
                quantized_value, True, probabilities, not swapped_scores, None, q.dtype,
                value_scale, probability_scale, None, False).transpose(-1, -2)
        else:
            output = torch.ops.hpu.fp8_gemm_v2(
                probabilities, swapped_scores, quantized_value, False, None, q.dtype,
                probability_scale, value_scale, None, False)
        return output.unsqueeze(0).unsqueeze(0)
    return forward


def install(mode):
    if mode == 'long-query':
        tiled = candidate('batch-query-softmax2d-av2d-1536')

        def long_query(q, k, v):
            return tiled(q, k, v) if q.shape[-2] >= 8192 else original_attention(q, k, v)

        attention._hpu_fp8_attention_av = long_query
    else:
        attention._hpu_fp8_attention_av = original_attention if mode == 'baseline' else candidate(mode)
