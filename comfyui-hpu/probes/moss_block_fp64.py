"""CPU FP64 mathematical block reference; never allocates an HPU tensor."""
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)
torch.set_num_threads(8)
root = Path(os.environ['MOSS_BLOCK_ROOT'])
capture = root / 'captures'
step = int(os.environ.get('MOSS_FP64_STEP', '0'))
condition = os.environ.get('MOSS_FP64_CONDITION')
prefix = f'{condition}-{step}' if condition else str(step)
results = []


def rms(x, weight):
    return x * (x.square().mean(dim=-1, keepdim=True) + 1e-6).rsqrt() * weight


def rope(x, pe):
    split = x.unflatten(-1, (2, -1)).movedim(-2, -1).unsqueeze(-2)
    return (pe[..., 0] * split[..., 0] + pe[..., 1] * split[..., 1]).movedim(-1, -2).flatten(-2)


for layer in (0, 20, 41):
    started = time.perf_counter()
    state = {k: v.double() for k, v in torch.load(capture / f'weights-{layer}.pt', weights_only=True).items()}
    data = torch.load(capture / f'input-{prefix}-{layer}.pt', weights_only=True)
    x = data['args'][0].double()
    mod = tuple(t.double() for t in data['args'][1])
    pe = data['args'][2].double()
    y = rms(x, state['norm1.weight']) * (1 + mod[1]) + mod[0]
    q, k, v = F.linear(y, state['proj.qkv.weight']).chunk(3, dim=-1)
    shape = (x.shape[0], x.shape[1], 24, 128)
    q = rope(rms(q.reshape(shape), state['q_norm.weight']), pe).transpose(1, 2)
    k = rope(rms(k.reshape(shape), state['k_norm.weight']), pe).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(x.shape)
    del q, k, v
    x = torch.addcmul(x, F.linear(y, state['proj.o.weight']), mod[2])
    y = rms(x, state['norm2.weight']) * (1 + mod[4]) + mod[3]
    hidden, gate = F.linear(y, state['proj.ffn_up_gate.weight']).chunk(2, dim=-1)
    y = (4 * torch.tanh(gate / 4) * torch.sigmoid(gate)) * (25 * torch.tanh(hidden / 25))
    out = torch.addcmul(x, F.linear(y, state['proj.ffn_down.weight']), mod[5])
    output_name = f'fp64-{prefix}-{layer}.pt' if condition else (f'fp64-{layer}.pt' if step == 0 else f'fp64-{step}-{layer}.pt')
    torch.save(out, root / output_name)
    record = {'layer': layer, 'step': step, 'seconds': time.perf_counter() - started, 'finite': bool(torch.isfinite(out).all())}
    results.append(record)
    (root / f'fp64-reference-{prefix}.json').write_text(json.dumps(results, indent=2))
    print(record, flush=True)
