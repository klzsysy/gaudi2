"""Separate packing and bound-check costs before changing the native pipeline."""
import argparse
import json
from pathlib import Path
import statistics
import time
import types

import torch
from qwen_d2_native_runtime import prepare


def candidate(q, k, v, query_buffer, key_buffer, value_buffer, kind):
    if kind == 'original':
        return prepare(q, k, v, query_buffer, key_buffer, value_buffer)
    mean = k.float().mean(dim=-2, keepdim=True)
    centered = (k.float()-mean).to(k.dtype)
    if kind != 'gates':
        query_buffer[:, :q.shape[-2]].copy_(q[0] * .0625)
        query_buffer[:, q.shape[-2]:].zero_()
        value_buffer[:, :v.shape[-2]].copy_(v[0])
        value_buffer[:, v.shape[-2]:].zero_()
        key_buffer[:, :, :k.shape[-2]].copy_(centered[0].transpose(-1, -2))
        key_buffer[:, :, k.shape[-2]:].zero_()
    if kind == 'packing':
        return mean[0].sum((-1, -2))
    maximum_value = v[0].abs().amax((-1, -2)).float()
    if kind == 'l1':
        query_l1 = q[0].float().abs().sum(-1)
        key_max = centered[0].abs().amax((-1, -2)).float()
        upper = query_l1.amax(-1) * key_max * (.0625 * 261.25 / 184.6649652337873)
        finite = query_l1.sum(-1) + mean[0].sum((-1, -2)) + v[0].float().sum((-1, -2))
    else:
        query_square = q[0].float().square().sum(-1)
        key_square = centered[0].float().square().sum(-1)
        upper = (query_square.amax(-1) * key_square.amax(-1)).sqrt() * (.0625 * 261.25 / 184.6649652337873)
        finite = query_square.sum(-1) + key_square.sum(-1) + v[0].float().sum((-1, -2))
    return torch.stack([upper, maximum_value, finite])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    options = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(32)
    saved = torch.load(options.input, weights_only=True)
    q, k, v = [saved[name].reshape(1, -1, 32, 128).transpose(1, 2).contiguous().to('hpu')
               for name in ('q', 'k', 'v')]
    buffers = (torch.empty(32, 8192, 128, device='hpu', dtype=q.dtype),
               torch.empty(32, 128, 9472, device='hpu', dtype=q.dtype),
               torch.empty(32, 9472, 128, device='hpu', dtype=q.dtype))
    compiled, results = {}, {}
    for kind in ('original', 'manual', 'l1', 'packing', 'gates'):
        function = types.FunctionType(candidate.__code__.replace(co_name='d2_prepare_' + kind), globals(), argdefs=(kind,))
        compiled[kind] = torch.compile(function, backend='hpu_backend', dynamic=None,
                                       options={'force_static_compile': True})
        for _ in range(3):
            output = compiled[kind](q, k, v, *buffers)
        torch.hpu.synchronize()
        result = output.cpu()
        samples = []
        for _ in range(7):
            started = time.perf_counter()
            for _ in range(10):
                output = compiled[kind](q, k, v, *buffers)
            torch.hpu.synchronize()
            samples.append((time.perf_counter()-started)/10)
        record = dict(seconds=statistics.median(samples), samples=samples, output=result.tolist())
        if kind not in ('packing', 'gates'):
            invalid = []
            for index in range(3):
                changed = [q.clone(), k.clone(), v.clone()]
                changed[index][0, 0, 0, 0] = torch.nan
                value = compiled[kind](*changed, *buffers).cpu()
                invalid.append(not bool(torch.isfinite(value[:, 0]).all()))
            assert all(invalid), (kind, invalid)
            record['nan_detected'] = invalid
        results[kind] = record
        (options.output / 'prepare-results.json').write_text(json.dumps(results, indent=2))
        print(kind, record['seconds'], flush=True)
    function = compiled['original']
    function(q, k, v, *buffers)
    torch.hpu.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.HPU]) as profile:
        for _ in range(5):
            function(q, k, v, *buffers)
        torch.hpu.synchronize()
    profile.profiler.export_chrome_trace(str(options.output / 'prepare.trace.json'))


if __name__ == '__main__':
    main()
