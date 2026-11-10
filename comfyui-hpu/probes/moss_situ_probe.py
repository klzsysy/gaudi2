"""Measure existing SiTU compilation without changing GEMM accumulation."""
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
capture = root / 'captures'
compiled = torch.compile(comfy.ops._situ_eager, backend='hpu_backend', fullgraph=True, dynamic=None)
records = []
for layer in (0, 20, 41):
    state = torch.load(capture / f'weights-{layer}.pt', weights_only=True)
    block = moss.Block(3072, 8192, 24, fused_qkv=True, dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
    block.load_state_dict(state)
    block.to('hpu')
    for step in (0, 12, 24):
        data = torch.load(capture / f'input-{step}-{layer}.pt', weights_only=True)
        x = data['args'][0].to('hpu')
        mod = tuple(t.to('hpu') for t in data['args'][1])
        pe = data['args'][2].to('hpu')
        y = moss.modulate(x, block.norm1, mod, 0, None)
        q, k, v = block.proj['qkv'](y).chunk(3, dim=-1)
        out = moss.attention(q, k, v, block.q_norm, block.k_norm, pe, 24, block.comfy_attention, {})
        x = moss.residual(x, block.proj['o'](out), mod, 2, None)
        y = block.proj['ffn_up_gate'](moss.modulate(x, block.norm2, mod, 3, None))
        expected = comfy.ops._situ_eager(y).cpu()
        actual = compiled(y).cpu()
        record = {'layer': layer, 'step': step, 'exact': bool(torch.equal(expected, actual)), 'max_abs': float((actual.float() - expected.float()).abs().max())}
        if step == 0:
            for name, function in [('eager', comfy.ops._situ_eager), ('compiled', compiled)]:
                samples = []
                for _ in range(5):
                    torch.hpu.synchronize()
                    start = time.perf_counter()
                    for _ in range(10):
                        function(y)
                    torch.hpu.synchronize()
                    samples.append((time.perf_counter() - start) / 10)
                record[name] = statistics.median(samples)
        records.append(record)
        print(json.dumps(record), flush=True)
        (root / 'situ-results.json').write_text(json.dumps(records, indent=2))
print('fallback', debug._get_fallback_op_count(), flush=True)
