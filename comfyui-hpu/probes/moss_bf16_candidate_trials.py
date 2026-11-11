"""Screen existing-op candidates on both current CFG shapes before whole requests."""
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True
import comfy.ops
from comfy.ldm.moss_image import model as moss
import moss_bf16_candidates as candidates
import habana_frameworks.torch.utils.debug as debug

torch.set_grad_enabled(False)
torch.set_num_threads(16)
root = Path(os.environ['MOSS_BF16_ROOT'])
state = torch.load(root / 'captures/weights-0.pt', weights_only=True)
block = moss.Block(3072, 8192, 24, fused_qkv=True, dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
block.load_state_dict(state)
block.to('hpu')
records = []
references = {}
for mode in os.environ.get('MOSS_BF16_CASES', 'product,compile-mod,compile-points,adaln,norm,rotary,native-all').split(','):
    torch._dynamo.reset()
    moss._compiled_block_ops.cache_clear()
    moss._compiled_attention.cache_clear()
    candidates.install(mode)
    for condition in (307, 23):
        data = torch.load(root / f'captures/input-{condition}-0-0.pt', weights_only=True)
        x = data['args'][0].to('hpu')
        mod = tuple(t.to('hpu') for t in data['args'][1])
        pe = data['args'][2].to('hpu')
        def run():
            return block(x.clone(), mod, pe, {})
        try:
            start = time.perf_counter()
            actual = run().cpu().float()
            cold = time.perf_counter() - start
            if mode == 'product':
                references[condition] = actual
            reference = references[condition]
            for _ in range(3):run()
            torch.hpu.synchronize()
            times = []
            for _ in range(5):
                start = time.perf_counter()
                for _ in range(10):run()
                torch.hpu.synchronize()
                times.append((time.perf_counter() - start) / 10)
            delta = actual-reference
            record = {'mode': mode, 'condition': condition, 'cold_s': cold,
                      'median_ms': statistics.median(times)*1000, 'samples_ms': [v*1000 for v in times],
                      'rrmse': float(delta.square().mean().sqrt()/reference.square().mean().sqrt()),
                      'max_abs': float(delta.abs().max()), 'finite': bool(actual.isfinite().all()),
                      'fallback': debug._get_fallback_op_count()}
            torch.save(actual, root / f'candidate-{mode}-{condition}.pt')
        except Exception:
            record = {'mode': mode, 'condition': condition, 'error': traceback.format_exc()}
        records.append(record)
        (root / os.environ.get('MOSS_CANDIDATE_RESULTS', 'candidate-trials.json')).write_text(json.dumps(records, indent=2))
        print(json.dumps(record), flush=True)
