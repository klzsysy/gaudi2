"""Screen sparse-parent path doubling against small dense matrix products."""
import json
import os
from pathlib import Path
import statistics
import time

import torch
from comfy.text_encoders import hpu_gated_delta as delta
import habana_frameworks.torch.utils.debug as debug


def paths(decay, parents):
    rows = decay.shape[-1]
    identity = torch.eye(rows, dtype=decay.dtype, device=decay.device)
    result = identity.expand(*decay.shape, rows)
    ancestor = parents
    factor = decay * (ancestor >= 0)
    for level in range((rows - 1).bit_length()):
        index = ancestor.clamp(min=0)
        result = result + factor.unsqueeze(-1) * result.index_select(-2, index)
        if level + 1 < (rows - 1).bit_length():
            factor = factor * factor.index_select(-1, index)
            ancestor = ancestor.index_select(0, index)
    initial = (result * (decay * (parents < 0)).unsqueeze(-2)).sum(-1)
    return result, initial, identity


if os.environ.get('MOSS_GDN_PATHS_INSTALL') == '1':
    delta._path_weights = paths
else:
    torch.set_grad_enabled(False)
    torch.set_num_threads(32)
    torch.manual_seed(43)
    root = Path(os.environ['MOSS_OPT_ROOT'])

    def core(state, pk, pv, pd, pb, q, k, v, d, b, program):
        committed = delta._commit(state, pk, pv, pd, pb)
        return committed, delta._tree_output(committed, q, k, v, d, b, program)

    original = delta._path_weights
    records = []
    for rows in [1, 4, 8]:
        for mode in ['chain', 'leaves', 'reordered']:
            batch, heads = 1, 32
            state = torch.randn(batch, heads, 128, 128, device='hpu') * .01
            pk = torch.randn(batch, heads, 128, 8, device='hpu') * .05
            pv = torch.randn(batch, heads, 128, 8, device='hpu') * .1
            pd = torch.rand(batch, 8, heads, device='hpu')
            pb = torch.rand(batch, 8, heads, device='hpu')
            q = torch.randn(batch, heads, 128, rows, device='hpu') * .05
            k = torch.randn_like(q) * .05
            v = torch.randn(batch, heads, 128, rows, device='hpu') * .1
            d = torch.rand(batch, rows, heads, device='hpu')
            b = torch.rand(batch, rows, heads, device='hpu')
            # Include exact zero-decay paths in every case.
            pd[:, 2] = 0
            d[:, -1] = 0
            order = list(range(rows)) if mode != 'reordered' else list(reversed(range(rows)))
            program = torch.tensor([row + (32 if mode != 'leaves' or index == 0 else 0)
                                    for index, row in enumerate(order)], device='hpu', dtype=torch.int64)
            inputs = (state, pk, pv, pd, pb, q, k, v, d, b, program)
            outputs, seconds = {}, {}
            for label, path_function in [('baseline', original), ('gather', paths)]:
                delta._path_weights = path_function
                torch._dynamo.reset()
                compiled = torch.compile(core, backend='hpu_backend', fullgraph=True, dynamic=False)
                for _ in range(4):
                    output = compiled(*inputs)
                torch.hpu.synchronize()
                outputs[label] = [x.cpu() for x in output]
                samples = []
                for _ in range(5):
                    start = time.perf_counter_ns()
                    for _ in range(30):
                        compiled(*inputs)
                    torch.hpu.synchronize()
                    samples.append((time.perf_counter_ns() - start) / 1e9 / 30)
                seconds[label] = statistics.median(samples)
            errors = []
            for actual, expected in zip(outputs['gather'], outputs['baseline']):
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=5e-5)
                errors.append(float((actual - expected).norm() / expected.norm()))
            record = dict(rows=rows, mode=mode, seconds=seconds, rrmse=errors,
                          fallback=debug._get_fallback_op_count())
            records.append(record)
            (root / 'gdn-paths.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(record), flush=True)
