"""Check whether HPU custom-op registration survives device initialization."""
import argparse
import json
from pathlib import Path

import torch
import comfy_kitchen
import habana_frameworks.torch


parser = argparse.ArgumentParser()
parser.add_argument('output', type=Path)
options = parser.parse_args()
calls = []


def marker(query, key, frequencies):
    calls.append(list(query.shape))
    return query.clone(), key.clone()


torch.set_grad_enabled(False)
torch.library.register_kernel('comfy_kitchen::apply_rope_split_half', 'hpu', marker)
query = torch.randn(1, 16, 4, 64, device='hpu', dtype=torch.bfloat16)
key = torch.randn(1, 4, 4, 64, device='hpu', dtype=torch.bfloat16)
frequencies = torch.randn(1, 1, 4, 32, 2, 2, device='hpu')
result = {}
for stage in ('early', 'late'):
    if stage == 'late':
        torch.library.register_kernel('comfy_kitchen::apply_rope_split_half', 'hpu', marker)
    for compiled in (False, True):
        before = len(calls)
        function = (torch.compile(comfy_kitchen.apply_rope_split_half, backend='hpu_backend', dynamic=False)
                    if compiled else comfy_kitchen.apply_rope_split_half)
        q, k = function(query, key, frequencies)
        torch.hpu.synchronize()
        result[f'{stage}-{compiled}'] = dict(calls=len(calls) - before,
                                             identity=bool(torch.equal(q, query) and torch.equal(k, key)))
options.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result))
