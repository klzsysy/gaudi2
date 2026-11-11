"""Compare captured GDN prefill calls with the same equations in CPU FP64."""
import inspect
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
from comfy.text_encoders.qwen35 import torch_chunk_gated_delta_rule
from comfy.text_encoders import hpu_gated_delta

torch.set_num_threads(8)
torch.set_grad_enabled(False)
root = Path(os.environ['MOSS_BF16_ROOT'])
namespace = {'torch': torch, 'F': F}
source = inspect.getsource(torch_chunk_gated_delta_rule).replace('torch.float32', 'torch.float64').replace('.float()', '.double()')
exec(source, namespace)
reference = namespace['torch_chunk_gated_delta_rule']
compiled = hpu_gated_delta.compiled_prefill(torch_chunk_gated_delta_rule)


def convert(value, device, dtype=None):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype or value.dtype)
    if isinstance(value, (tuple, list)):
        return type(value)(convert(v, device, dtype) for v in value)
    if isinstance(value, dict):
        return {k: convert(v, device, dtype) for k, v in value.items()}
    return value


rows = []
for path in sorted((root / 'prefill-capture').glob('prefill-*.pt')):
    data = torch.load(path, weights_only=True, map_location='cpu')
    expected = reference(*convert(data['args'], 'cpu', torch.float64), **convert(data['kwargs'], 'cpu', torch.float64))
    values, kwargs = convert(data['args'], 'hpu'), convert(data['kwargs'], 'hpu')
    for name, function in [('eager', torch_chunk_gated_delta_rule), ('compiled-bucketed', compiled)]:
        actual = function(*values, **kwargs)
        errors = []
        for a, b in zip(actual, expected):
            if b is None:
                assert a is None
                errors.append(None)
                continue
            a = a.cpu().double()
            delta = a - b
            errors.append(dict(rrmse=float(delta.square().mean().sqrt()/b.square().mean().sqrt()),
                               max_abs=float(delta.abs().max()), finite=bool(a.isfinite().all())))
        rows.append(dict(capture=path.name, mode=name, errors=errors))
        (root / 'prefill-fp64.json').write_text(json.dumps(rows, indent=2))
        print(json.dumps(rows[-1]), flush=True)
