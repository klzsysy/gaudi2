"""Test rectangular Qwen attention on the locally rebuilt SRAM runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
from habana_frameworks.torch.utils.experimental import _data_ptr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--head-start', type=int, default=0)
    parser.add_argument('--parts', type=int, default=4, choices=(1, 2, 4))
    parser.add_argument('--group', type=int, default=16, choices=(16, 32, 64))
    options = parser.parse_args()
    source_name = 'native-source' if options.parts == 4 else f'native-parts{options.parts}'
    source = options.root / source_name / 'mme-direct-attention/attention'
    os.environ['H3DIRECT_ROOT'] = str(source)
    os.environ['H3DIRECT_POW2Q'] = '1'
    sys.path.insert(0, str(source))
    import h3_direct_attention as direct
    direct.GROUP = options.group
    torch.set_grad_enabled(False)
    torch.set_num_threads(32)
    saved = torch.load(options.input, weights_only=False)
    tensors = [saved[name].reshape(1, -1, saved['heads'], 128).transpose(1, 2)
               [:, options.head_start:options.head_start + options.heads].contiguous() for name in ('q', 'k', 'v')]
    q, k, v = [tensor.to('hpu') for tensor in tensors]
    query_length, key_length = q.shape[-2], k.shape[-2]
    capacity = ((key_length + 127) // 128) * 128
    if options.parts == 4 and capacity % 512 == 0:
        capacity += 128
    quantum = options.parts * 128
    first = (capacity + quantum - 1) // quantum * 128
    group_rows = options.group * 128
    blocks = (query_length + group_rows - 1) // group_rows * options.group
    assert key_length > (options.parts - 1) * first
    direct.geometry = lambda keys: (first, blocks, (options.parts - 1) * first + 1)
    centered = (k.float() - k.float().mean(dim=-2, keepdim=True)).to(k.dtype)
    qs = q[0] * .0625
    gates = direct.head_gates(qs, centered[0], v[0], direct.SCORE_FACTOR, key_length)
    print('numeric gates', gates, flush=True)
    assert all(gates), 'Input is outside this bounded-exp prototype'
    check = torch.linspace(.25, 4, 4096).reshape(64, 64).to('hpu')
    before = {name: function(check).cpu() for name, function in
              [('exp', torch.exp), ('sigmoid', torch.sigmoid), ('tanh', torch.tanh), ('rsqrt', torch.rsqrt)]}
    started = time.perf_counter()
    plan = direct._Plan(False, options.heads, capacity, key_length)
    prepare_seconds = time.perf_counter() - started

    def stage():
        plan.qh[:, :query_length].copy_(qs)
        plan.vh[:, :key_length].copy_(v[0])
        for part in range(options.parts):
            start = part * first
            width = min(first, capacity - start)
            valid = min(width, key_length - start)
            target = plan.kth[:, start*128:(start+width)*128].view(options.heads, 128, width)
            target[:, :, :valid].copy_(centered[0, :, start:start+valid].transpose(1, 2))

    def compute(output):
        plan.run_heads_stream(list(range(options.heads)), _data_ptr(output))

    stage()
    output = torch.full((options.heads, blocks*128, 128), torch.nan, dtype=q.dtype, device='hpu')
    torch.hpu.synchronize()
    compute(output)
    actual = output[:, :query_length].cpu()
    assert actual.isfinite().all(), 'Non-finite native output'
    rows = torch.linspace(0, query_length-1, 33).long()
    cpu_q, cpu_k, cpu_v = [tensor[0].double() for tensor in tensors]
    reference = ((cpu_q[:, rows] @ cpu_k.transpose(-1, -2)) / 128 ** .5).softmax(-1) @ cpu_v
    sample = actual[:, rows].double()
    errors = (sample-reference).square().sum((1, 2)).sqrt()/reference.square().sum((1, 2)).sqrt()
    times, stage_times = [], []
    for _ in range(7):
        started = time.perf_counter()
        stage()
        torch.hpu.synchronize()
        stage_times.append(time.perf_counter()-started)
        started = time.perf_counter()
        compute(output)
        times.append(time.perf_counter()-started)
    assert torch.equal(actual, output[:, :query_length].cpu()), 'Native repeated output changed'
    special_equal = {name: torch.equal(function(check).cpu(), before[name]) for name, function in
                     [('exp', torch.exp), ('sigmoid', torch.sigmoid), ('tanh', torch.tanh), ('rsqrt', torch.rsqrt)]}
    assert all(special_equal.values()), special_equal
    result = dict(input=str(options.input), heads=options.heads, head_start=options.head_start,
                  query=query_length, keys=key_length, capacity=capacity, parts=options.parts, first_part=first,
                  query_blocks=blocks, group=options.group, prepare_seconds=prepare_seconds, native_times=times,
                  native_seconds=statistics.median(times), staging_times=stage_times,
                  staging_seconds=statistics.median(stage_times),
                  fp64_rrmse=float((sample-reference).norm()/reference.norm()), head_rrmse=errors.tolist(),
                  special_ops_equal=special_equal, repeated_equal=True)
    result['runtime_sha256'] = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in
                               ('build/libcontext_bridge.so', 'runtime_quad.hpp', 'h3_direct_attention.py',
                                'build/normalize_quad_fixed1024.bin', 'build/exp_f16_tile_fixed0_pow2q_g15pad.bin')}
    torch.save(actual, options.output / 'output.pt')
    (options.output / 'native-result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    torch.hpu.synchronize()
    plan.release()
    assert torch.equal(torch.exp(check).cpu(), before['exp'])


if __name__ == '__main__':
    main()
