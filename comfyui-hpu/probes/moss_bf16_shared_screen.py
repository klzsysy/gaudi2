"""Evaluate shared block compilation and query batching on real BF16 SFT inputs."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.use_pytorch_cross_attention = True
import comfy.ops
from comfy.ldm.moss_image import model as moss
from habana_frameworks.torch.utils.debug import _get_fallback_op_count


parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('--captures', type=Path, required=True)
parser.add_argument('--pattern', default='input-*-12-0.pt')
parser.add_argument('--modes', default='baseline,whole-block,rank3,batch-query-512,batch-query-1536,baseline')
options = parser.parse_args()
options.root.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
torch.set_num_threads(32)
original_sdpa = comfy.ops.scaled_dot_product_attention
original_modulate = moss.modulate
original_residual = moss.residual


def query_batched(mode):
    tile = int(mode.removeprefix('batch-query-')) if mode.startswith('batch-query-') else None

    def forward(q, k, v, *values, **kwargs):
        if mode == 'rank3':
            return original_sdpa(q.squeeze(0), k.squeeze(0), v.squeeze(0), *values, **kwargs).unsqueeze(0)
        length, width = q.shape[-2:]
        query = torch.nn.functional.pad(q[0, 0], (0, 0, 0, -length % tile)).reshape(-1, 1, tile, width)
        batches = query.shape[0]
        output = original_sdpa(query, k.expand(batches, -1, -1, -1), v.expand(batches, -1, -1, -1), *values, **kwargs)
        return output.reshape(-1, width)[:length].unsqueeze(0).unsqueeze(0)
    return forward


records = []
for source in sorted(options.captures.glob(options.pattern)):
    data = torch.load(source, weights_only=True)
    block = moss.Block(3072, 8192, 24, fused_qkv=data['fused_qkv'], dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
    block.load_state_dict(torch.load(options.captures / f"weights-{data['layer']}.pt", weights_only=True))
    block.to('hpu')
    x, mod, pe = data['args']
    x, mod, pe = x.to('hpu'), tuple(t.to('hpu') for t in mod), pe.to('hpu')
    settings = dict(_moss_hpu_blockwise_rope=moss.hpu.blockwise_coefficients(pe))
    reference = None
    for mode in options.modes.split(','):
        whole_block = mode.startswith('whole-block')
        comfy.ops.scaled_dot_product_attention = original_sdpa if mode == 'baseline' or whole_block else query_batched(mode)
        moss.modulate, moss.residual = original_modulate, original_residual
        if mode == 'whole-block-preserve':
            moss.modulate = torch.compiler.disable(original_modulate)
            moss.residual = torch.compiler.disable(original_residual)
        torch._dynamo.reset()
        moss._compiled_block_ops.cache_clear()
        moss._compiled_block_forward.cache_clear()
        function = moss._compiled_block_forward() if whole_block else moss.Block.forward
        for _ in range(3):
            output = function(block, x.clone(), mod, pe, settings)
        torch.hpu.synchronize()
        actual = output.cpu()
        if reference is None:
            assert mode == 'baseline'
            reference = actual
        delta = actual.float() - reference.float()
        samples = []
        for _ in range(7):
            start = time.perf_counter_ns()
            for _ in range(10):
                output = function(block, x.clone(), mod, pe, settings)
            torch.hpu.synchronize()
            samples.append((time.perf_counter_ns() - start) / 1e9 / 10)
        record = dict(source=source.name, mode=mode, equal=torch.equal(actual, reference),
                      differing=int((actual != reference).sum()), rrmse=float(delta.norm() / reference.float().norm()),
                      maximum=float(delta.abs().max()), finite=bool(actual.isfinite().all()),
                      seconds=statistics.median(samples), samples=samples, fallback=_get_fallback_op_count())
        records.append(record)
        (options.root / 'bf16-shared-screen.json').write_text(json.dumps(records, indent=2))
        print(json.dumps(record), flush=True)
        assert record['finite'] and not record['fallback'], record
