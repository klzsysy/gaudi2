"""Capture baseline deferred decode inputs and mutated outputs at selected steps."""
import os
from pathlib import Path
import torch
from comfy.text_encoders import hpu_gated_delta


def install():
    directory = Path(os.environ['MOSS_RESULT_DIR'])/'delta-captures'
    directory.mkdir(parents=True,exist_ok=True)
    conv,delta = hpu_gated_delta.compiled_ops()
    calls = 0
    conv_calls = 0
    def capture_conv(*values):
        nonlocal conv_calls
        step,layer = divmod(conv_calls,24)
        conv_calls += 1
        selected = step in (0,1,12,64,128) and layer in (0,12,23)
        if selected:
            before = [v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v for v in values]
        output = conv(*values)
        if selected:
            torch.save(dict(args=before,after=[v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v for v in values]),directory/f'conv-{step}-{layer}.pt')
        return output
    def capture(*values):
        nonlocal calls
        step,layer = divmod(calls,24)
        calls += 1
        selected = step in (0,1,12,64,128) and layer in (0,12,23)
        if selected:
            before = [v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v for v in values]
        output = delta(*values)
        if selected:
            torch.save(dict(args=before,output=output.cpu(),state=values[5].cpu(),gates=values[13].cpu(),sumsq=values[14].cpu()),directory/f'delta-{step}-{layer}.pt')
        return output
    hpu_gated_delta.compiled_ops = lambda:(capture_conv,capture)
