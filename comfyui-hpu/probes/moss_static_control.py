"""Specialize existing Kitchen calls with already-known host control scalars."""
import torch
import comfy_kitchen
from comfy.text_encoders import qwen35, hpu_gated_delta


def install(mode):
    torch._dynamo.config.recompile_limit = 64  # Isolated experiment; cold cost is reported.
    original_forward = qwen35.GatedDeltaNet.forward
    original_ops = hpu_gated_delta.compiled_ops()
    current = None

    def control(ctl, pending, parity):
        return torch.cat((torch.tensor([pending, parity], dtype=ctl.dtype, device=ctl.device), ctl[2:]))

    def conv(*values, pending, parity):
        return comfy_kitchen.deltanet_conv_step_deferred(*values[:-1], control(values[-1], pending, parity))

    def delta(*values, pending, parity):
        return comfy_kitchen.gated_delta_decode_deferred(*values[:-1], control(values[-1], pending, parity))

    compiled_conv = torch.compile(conv, backend='hpu_backend', dynamic=False, options={'force_static_compile': True})
    compiled_delta = torch.compile(delta, backend='hpu_backend', dynamic=False, options={'force_static_compile': True})

    def conv_call(*values):
        if mode in ('both', 'conv'):
            return compiled_conv(*values, pending=current[0], parity=current[1])
        return original_ops[0](*values)

    def delta_call(*values):
        if mode in ('both', 'delta'):
            return compiled_delta(*values, pending=current[0], parity=current[1])
        return original_ops[1](*values)

    def forward(module, x, past_key_value=None, **kwargs):
        nonlocal current
        if past_key_value is not None and past_key_value.decode_step(x.shape[1]):
            tracker = past_key_value.ctl_tracker
            parity = tracker['parity']
            current = (int(tracker['host'][parity][0]), parity)
        return original_forward(module, x, past_key_value=past_key_value, **kwargs)

    qwen35.GatedDeltaNet.forward = forward
    hpu_gated_delta.compiled_ops = lambda: (conv_call, delta_call)
