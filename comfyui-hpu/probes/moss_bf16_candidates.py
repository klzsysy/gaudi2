"""Investigation-only BF16 normalization/rotary candidates using vendor ops."""
import torch
import comfy.ops
import comfy.quant_ops
from comfy.ldm.moss_image import model as moss
from comfy_kitchen.backends.eager.rope import apply_rope_split_half1


original_adaln = comfy.quant_ops.ck.rms_adaln
original_rope = comfy.quant_ops.ck.rms_rope_split_half
original_modulation = moss.modulated_rms_norm
original_residual = moss.gated_residual


def rms(x, weight, eps):
    return torch.ops.hpu.rms_norm(x, weight, eps)[0]


def adaln(x, scale, shift, eps=1e-6):
    normalized = rms(x, torch.ones(x.shape[-1], device=x.device, dtype=x.dtype), eps)
    return normalized * (1 + scale) + shift


def native_rope(x, pe):
    cos = torch.cat((pe[..., 0, 0], pe[..., 1, 1]), dim=-1)
    sin = torch.cat((-pe[..., 0, 1], pe[..., 1, 0]), dim=-1)
    return torch.ops.hpu.rotary_pos_embedding(x.float(), sin, cos, None, 0, 0).to(x.dtype)


def norm_rope(q, k, pe, q_scale, k_scale=None, epsilon=1e-6, rot_dim=0):
    k_scale = q_scale if k_scale is None else k_scale
    return (apply_rope_split_half1(rms(q, q_scale, epsilon), pe),
            apply_rope_split_half1(rms(k, k_scale, epsilon), pe))


def all_native_rope(q, k, pe, q_scale, k_scale=None, epsilon=1e-6, rot_dim=0):
    k_scale = q_scale if k_scale is None else k_scale
    return native_rope(rms(q, q_scale, epsilon), pe), native_rope(rms(k, k_scale, epsilon), pe)


def rotary_only(q, k, pe, q_scale, k_scale=None, epsilon=1e-6, rot_dim=0):
    k_scale = q_scale if k_scale is None else k_scale
    return (native_rope(torch.nn.functional.rms_norm(q, (q.shape[-1],), q_scale, epsilon), pe),
            native_rope(torch.nn.functional.rms_norm(k, (k.shape[-1],), k_scale, epsilon), pe))


def rounded_rms(x, weight, eps):
    return rms(x, torch.ones(x.shape[-1], device=x.device, dtype=x.dtype), eps) * weight


def rounded_norm_rope(q, k, pe, q_scale, k_scale=None, epsilon=1e-6, rot_dim=0):
    k_scale = q_scale if k_scale is None else k_scale
    return native_rope(rounded_rms(q, q_scale, epsilon), pe), native_rope(rounded_rms(k, k_scale, epsilon), pe)


def install(mode):
    comfy.quant_ops.ck.rms_adaln = adaln if mode in ('adaln', 'native-all', 'adaln-rotary', 'native-rounded', 'adaln-rotary-compiled') else original_adaln
    comfy.quant_ops.ck.rms_rope_split_half = {'norm': norm_rope, 'rotary': rotary_only, 'native-all': all_native_rope,
                                            'adaln-rotary': rotary_only, 'adaln-rotary-compiled': rotary_only,
                                            'native-rounded': rounded_norm_rope}.get(mode, original_rope)
    if mode == 'adaln-rotary-compiled':
        comfy.quant_ops.ck.rms_adaln = torch.compile(adaln, backend='hpu_backend', dynamic=None,
                                                     options={'force_static_compile': True})
    moss.modulated_rms_norm = original_modulation
    moss.gated_residual = original_residual
    if mode in ('compile-mod', 'compile-points'):
        moss.modulated_rms_norm = torch.compile(original_modulation, backend='hpu_backend', dynamic=None,
                                                 options={'force_static_compile': True})
    if mode == 'compile-points':
        moss.gated_residual = torch.compile(original_residual, backend='hpu_backend', dynamic=None,
                                             options={'force_static_compile': True})
