"""Validate and time SRAM recipes using the framework's default Synapse stream."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import statistics
import time

import torch
from habana_frameworks.torch.utils import _debug_eager_C
from habana_frameworks.torch.utils.experimental import _compute_stream, _data_ptr


def address(tensor):
    return _data_ptr(tensor) + tensor.storage_offset() * tensor.element_size()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--queries', type=int, default=512)
    parser.add_argument('--keys', type=int, default=256)
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--tile', type=int, default=128)
    parser.add_argument('--slots', type=int, default=1)
    parser.add_argument('--alias', action='store_true')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--no-stats', action='store_true')
    parser.add_argument('--poly', action='store_true')
    parser.add_argument('--exp', action='store_true')
    parser.add_argument('--unroll', action='store_true')
    parser.add_argument('--core', action='store_true')
    options = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(32)
    torch.manual_seed(428)
    if options.input:
        saved = torch.load(options.input, weights_only=True)
        q, k, v = [saved[name].reshape(1, -1, 32, 128).transpose(1, 2)[:, :options.heads].contiguous().to('hpu')
                   for name in ('q', 'k', 'v')]
        q = q[:, :, :options.queries].contiguous()
        k = (k.float()-k.float().mean(-2, keepdim=True)).bfloat16()
        k, v = k[:, :, :options.keys].contiguous(), v[:, :, :options.keys].contiguous()
    else:
        q = torch.randn(1, options.heads, options.queries, 128, dtype=torch.bfloat16).to('hpu')
        k, v = [torch.randn(1, options.heads, options.keys, 128, dtype=torch.bfloat16).to('hpu') for _ in range(2)]
    q = q * 128 ** -.5
    output = torch.full_like(q, torch.nan)
    maximum = torch.empty(1, options.heads, options.queries, 1, device='hpu', dtype=q.dtype)
    inverse = torch.empty_like(maximum)
    torch.hpu.synchronize()
    lib = C.CDLL(str(options.root / 'rmw/librmw.so'))
    lib.build_attention.argtypes = [C.c_uint] * 5 + [C.c_int, C.c_int, C.c_int, C.c_char_p]
    lib.build_attention.restype = C.c_void_p
    lib.workspace_bytes.argtypes, lib.workspace_bytes.restype = [C.c_void_p], C.c_uint64
    lib.launch_attention.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(C.c_uint64), C.c_uint64, C.c_int]
    lib.destroy_attention.argtypes = [C.c_void_p]
    started = time.perf_counter()
    custom = options.poly or options.exp or options.core
    kind = 2 + int(options.exp) + 2 * int(options.unroll) if custom else int(options.fast)
    if options.core:
        kind = 6 + int(options.unroll)
    plan = lib.build_attention(options.queries, options.keys, options.heads, options.tile, options.slots,
                               options.alias, kind,
                               not options.no_stats, str(options.output / 'recipe').encode())
    assert plan, 'RMW recipe compilation failed'
    cold = time.perf_counter()-started
    workspace_size = lib.workspace_bytes(plan)
    workspace = torch.empty(max(256, workspace_size), device='hpu', dtype=torch.uint8)
    tensors = (q, k, v, output, maximum, inverse)
    pointers = (C.c_uint64 * 6)(*[address(tensor) for tensor in tensors])
    _debug_eager_C.join_pending_pipeline_threads()
    stream = _compute_stream()
    assert stream

    def launch(reduced=False):
        status = lib.launch_attention(plan, stream, pointers, address(workspace), reduced)
        assert status == 0, ('launch', status, reduced)

    try:
        launch()
        torch.hpu.synchronize()
        actual = output.cpu()
        assert actual.isfinite().all()
        rows = torch.linspace(0, options.queries-1, 17).long()
        cq, ck, cv = [tensor.cpu().double() for tensor in (q, k, v)]
        scores = cq[:, :, rows] @ ck.transpose(-1, -2)
        reference = scores.softmax(-1) @ cv
        sample = actual[:, :, rows].double()
        error = float((sample-reference).norm()/reference.norm())
        assert error < (.08 if options.fast else .03), error
        reduced_status = lib.launch_attention(plan, stream, pointers, address(workspace), 1)
        torch.hpu.synchronize()
        assert torch.equal(output.cpu(), actual)
        samples = []
        for _ in range(7):
            started = time.perf_counter()
            for _ in range(5):
                launch(reduced_status == 0)
            torch.hpu.synchronize()
            samples.append((time.perf_counter()-started)/5)
        assert torch.equal(output.cpu(), actual)
        record = dict(queries=options.queries, keys=options.keys, heads=options.heads, tile=options.tile,
                      slots=options.slots, alias=options.alias, fast=options.fast, poly=options.poly,
                      exp=options.exp, core=options.core, unroll=options.unroll,
                      pipeline='QWEN_RMW_PIPELINE' in os.environ,
                      retained=not options.no_stats and not custom, compile_seconds=cold,
                      workspace_bytes=workspace_size, reduced_binding_status=reduced_status,
                      fp64_rrmse=error, seconds=statistics.median(samples), samples=samples)
        if options.fast and not options.no_stats:
            m, inv = maximum.cpu(), inverse.cpu()
            assert m.isfinite().all() and inv.isfinite().all() and (inv > 0).all()
            record['maximum_abs_error'] = float((m[:, :, rows, 0].double()-scores.amax(-1)).abs().max())
            reference_inverse = (scores-scores.amax(-1, keepdim=True)).exp().sum(-1).reciprocal()
            record['inverse_rrmse'] = float((inv[:, :, rows, 0].double()-reference_inverse).norm()/reference_inverse.norm())
            record['inverse128_rrmse'] = float((inv[:, :, rows, 0].double()*128-reference_inverse).norm()/reference_inverse.norm())
        torch.save(actual, options.output / 'output.pt')
        (options.output / 'rmw-result.json').write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)
    finally:
        torch.hpu.synchronize()
        lib.destroy_attention(plan)


if __name__ == '__main__':
    main()
