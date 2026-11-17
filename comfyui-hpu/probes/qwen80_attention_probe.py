"""Check decode valid-length alignment and compare real GQA layouts."""
import json
from pathlib import Path
import statistics
import sys
import time
import torch
import habana_frameworks.torch
from habana_frameworks.torch.hpex.kernels import FusedSDPA


def attention(query, key, value, length, mode):
    if mode == 'valid':
        return FusedSDPA.apply(query, key, value, None, 0., True, None, 'fp32', False, length, 'right')
    mask = (torch.arange(key.shape[2], device=query.device).view(1, 1, 1, -1) >= length.view(-1, 1, 1, 1))
    mask = torch.zeros(mask.shape, dtype=query.dtype, device=query.device).masked_fill(mask, -float('inf'))
    if mode == 'grouped':
        groups = query.reshape(1, 4, 4, 256)
        scores = torch.matmul(groups.float(), key.float().transpose(-1, -2)) * (256 ** -.5)
        probs = torch.softmax(scores + mask.float(), dim=-1)
        return torch.matmul(probs, value.float()).reshape_as(query).to(query.dtype)
    return FusedSDPA.apply(query, key, value, mask, 0., False, None, 'fp32', mode == 'recompute')


torch.set_grad_enabled(False)
torch.set_num_threads(32)
output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
compiled = torch.compile(attention, backend='hpu_backend', fullgraph=True, dynamic=False,
                         options={'force_static_compile': True})
rows = []
for length in (2804, 4097, 4712):
    torch.manual_seed(42)
    query = torch.randn(1, 16, 1, 256, device='hpu', dtype=torch.bfloat16)
    key = torch.randn(1, 4, 6899, 256, device='hpu', dtype=torch.bfloat16)
    value = torch.randn_like(key)
    limit = torch.tensor([length], device='hpu', dtype=torch.int32)
    reference = compiled(query, key, value, limit, 'baseline').cpu().float()
    for mode in ('baseline', 'valid', 'recompute', 'grouped'):
        try:
            actual = compiled(query, key, value, limit, mode).cpu().float()
            seconds = []
            for repeat in range(6):
                torch.hpu.synchronize()
                start = time.perf_counter()
                for _ in range(100):
                    compiled(query, key, value, limit, mode)
                torch.hpu.synchronize()
                seconds.append((time.perf_counter() - start) / 100)
            rows.append(dict(length=length, mode=mode, seconds=statistics.median(seconds[1:]),
                             exact=torch.equal(actual, reference),
                             rrmse=((actual-reference).square().mean()/reference.square().mean()).sqrt().item()))
        except Exception as error:
            rows.append(dict(length=length, mode=mode, error=str(error)))
        (output / 'results.json').write_text(json.dumps(rows, indent=2))
        print(rows[-1], flush=True)
