"""Compare vendor static recipes with automatic Dynamo shape adaptation."""
import json
import os
from pathlib import Path
import statistics
import time

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True
import comfy.ops
from comfy.ldm.moss_image import model as moss
import habana_frameworks.torch.utils.debug as debug

torch.set_grad_enabled(False)
torch.set_num_threads(16)
root = Path(os.environ['MOSS_BLOCK_ROOT'])
state = torch.load(root / 'captures/weights-0.pt', weights_only=True)
data = torch.load(root / 'captures/input-0-0.pt', weights_only=True)
block = moss.Block(3072, 8192, 24, fused_qkv=True, dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
block.load_state_dict(state)
block.to('hpu')
x, pe = data['args'][0].to('hpu'), data['args'][2].to('hpu')
mod = tuple(v.to('hpu') for v in data['args'][1])
records = {}
dynamic_outputs = {}
for force_static in (False, True):
    torch._dynamo.reset()
    options = {'force_static_compile': force_static}
    compiled = (torch.compile(moss.Block._forward_attention, backend='hpu_backend', dynamic=None, options=options),
                torch.compile(moss.Block._forward_ffn, backend='hpu_backend', dynamic=None, options=options))
    moss._compiled_block_ops = lambda: compiled
    measurements = []
    for length in (4408, 4119, 4401, 4399, 4147, 1216, 1047, 257, 129, 65, 33, 17, 18, 4408, 4119):
        part = x[:, :length].clone()
        positions = pe[:, :length].clone()
        args.disable_comfy_compiler = True
        reference = block(part.clone(), mod, positions).cpu().float()
        args.disable_comfy_compiler = False
        torch.hpu.synchronize()
        work = part.clone()
        started = time.perf_counter()
        actual = block(work, mod, positions)
        torch.hpu.synchronize()
        cold = time.perf_counter() - started
        actual_cpu = actual.cpu().float()
        delta = actual_cpu - reference
        if not force_static:
            dynamic_outputs[length] = actual_cpu
        samples = []
        for _ in range(5):
            torch.hpu.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                work.copy_(part)
                block(work, mod, positions)
            torch.hpu.synchronize()
            samples.append((time.perf_counter() - start) / 10)
        record = {'length': length, 'first_seconds': cold, 'median': statistics.median(samples),
                  'relative_rmse': float(delta.square().mean().sqrt() / reference.square().mean().sqrt()),
                  'finite': bool(torch.isfinite(delta).all()), 'in_use': torch.hpu.memory_stats()['InUse']}
        if force_static:
            record['max_vs_dynamic'] = float((actual_cpu - dynamic_outputs[length]).abs().max())
        measurements.append(record)
        print(force_static, record, flush=True)
    records[str(force_static)] = measurements
    (root / 'shape-options.json').write_text(json.dumps(records, indent=2))
print('fallback', debug._get_fallback_op_count(), flush=True)
