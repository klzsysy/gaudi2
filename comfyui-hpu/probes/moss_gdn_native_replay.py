"""Compare complete deferred decode on recorded model activations."""
import json
import os
from pathlib import Path

import torch
from comfy.text_encoders import hpu_gated_delta as delta
import habana_frameworks.torch.utils.debug as debug

baseline_function = delta._decode_deferred
from native_gdn.install import decode

torch.set_grad_enabled(False)
torch.set_num_threads(32)
root = Path(os.environ['MOSS_OPT_ROOT'])
captures = root.parent / 'performance-2026-11-11-bf16-next/baseline-capture/delta-captures'
functions = {label: torch.compile(function, backend='hpu_backend', fullgraph=True, dynamic=False)
             for label, function in [('baseline', baseline_function), ('candidate', decode)]}
records = []
for path in sorted(captures.glob('delta-*.pt')):
    source = torch.load(path, weights_only=True)
    outputs = {}
    for label, function in functions.items():
        values = [value.to('hpu').clone() if isinstance(value, torch.Tensor) else value
                  for value in source['args']]
        # Match the existing four-row boundary materialization.
        if values[0].shape[1] == 4:
            values[0] = values[0].clone()
        output = function(*values)
        outputs[label] = [output.cpu(), values[5].cpu(), values[13].cpu(), values[14].cpu()]
    errors = []
    for actual, expected in zip(outputs['candidate'], outputs['baseline']):
        errors.append(dict(rrmse=float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)),
                           max_absolute=float((actual.float() - expected.float()).abs().max()),
                           finite=bool(actual.isfinite().all())))
    record = dict(file=path.name, shape=list(source['args'][0].shape), errors=errors,
                  fallback=debug._get_fallback_op_count())
    records.append(record)
    (root / 'gdn-real-replay.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(record), flush=True)
