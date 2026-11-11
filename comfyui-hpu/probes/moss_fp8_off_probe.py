"""Diagnose single-row DeltaNet recipe inputs after FP8 projections."""
from pathlib import Path
import runpy
import os
import torch
from comfy.text_encoders import hpu_gated_delta

original = hpu_gated_delta.compiled_ops
seen = set()
def operations():
    conv, decode = original()
    def checked(x, *values, **kwargs):
        key = (tuple(x.shape), str(x.dtype))
        if key not in seen:
            seen.add(key)
            print('DELTA_INPUTS', [(type(v).__name__, list(v.shape), str(v.dtype), list(v.stride())) for v in (x, *values) if isinstance(v, torch.Tensor)], flush=True)
        if x.shape[1] == 1 and os.environ.get('MOSS_SINGLE_CLONE') == '1':
            x = x.clone()
        return decode(x, *values, **kwargs)
    return conv, checked
hpu_gated_delta.compiled_ops = operations
runpy.run_path(str(Path(__file__).with_name('moss_gdn_experiment.py')), run_name='__main__')
