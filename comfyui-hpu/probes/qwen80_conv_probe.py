"""Validate installed native causal convolution against current BF16 boundaries."""
import json
from pathlib import Path
import sys
import torch
import habana_frameworks.torch

torch.set_grad_enabled(False)
torch.set_num_threads(32)
output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
rows = []
for dtype in (torch.bfloat16, torch.float32):
    for width in (3,):
        for activation in (False, True):
            torch.manual_seed(18)
            state = torch.randn(1, 256, width, device='hpu', dtype=dtype)
            value = torch.randn(1, 256, device='hpu', dtype=dtype)
            weight = torch.randn(256, 4, device='hpu', dtype=dtype)
            window = torch.cat((state[..., -3:], value.unsqueeze(-1)), dim=-1)
            reference = (window.float().flip(-1) * weight.float().flip(-1)).sum(-1).to(dtype)
            if activation:
                rounded = reference.float()
                reference = (rounded / (1. + torch.exp(-rounded))).to(dtype)
            try:
                result, updated = torch.ops.hpu.causal_conv1d_update(
                    value.unsqueeze(1), state.transpose(1, 2).contiguous(), weight.t().contiguous(),
                    None, activation=activation)
                result = result[:, 0]
                updated = updated.transpose(1, 2)
                actual, reference = result.cpu().float(), reference.cpu().float()
                rows.append(dict(dtype=str(dtype), width=width, activation=activation,
                                 rrmse=((actual-reference).square().mean()/reference.square().mean()).sqrt().item(),
                                 exact=torch.equal(actual, reference),
                                 state_exact=torch.equal(updated.cpu(), window[..., -width:].cpu())))
            except Exception as error:
                rows.append(dict(dtype=str(dtype), width=width, activation=activation, error=str(error)))
            print(rows[-1], flush=True)
            (output / 'results.json').write_text(json.dumps(rows, indent=2))
