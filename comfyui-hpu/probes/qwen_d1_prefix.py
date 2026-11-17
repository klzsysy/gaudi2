"""Measure Qwen prefix materialization, stable buffers, and segmented KV."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from qwen_d1_screen import attention, compile_candidates


def fused(q, prefix_k, prefix_v, k, v):
    return attention(q, torch.cat((prefix_k, k), 1), torch.cat((prefix_v, v), 1), 32,
                     kind='split', mode='fast', centered=True)


def segmented(q, prefix_k, prefix_v, k, v):
    q, prefix_k, prefix_v, k, v = [tensor.reshape(1, tensor.shape[1], 32, 128).transpose(1, 2)
                                  for tensor in (q, prefix_k, prefix_v, k, v)]
    mean = (prefix_k.float().sum(-2, keepdim=True) + k.float().sum(-2, keepdim=True)) / (prefix_k.shape[-2] + k.shape[-2])
    prefix_k = (prefix_k.float() - mean).to(k.dtype)
    k = (k.float() - mean).to(k.dtype)
    q = q.contiguous()
    first, maximum1, inverse1 = torch.ops.hpu.sdpa_recomp_fwd(
        q, prefix_k.contiguous(), prefix_v.contiguous(), None, 0.0, 128 ** -.5,
        False, True, 'fast', None, 'right')[:3]
    second, maximum2, inverse2 = torch.ops.hpu.sdpa_recomp_fwd(
        q, k.contiguous(), v.contiguous(), None, 0.0, 128 ** -.5, False, True, 'fast', None, 'right')[:3]
    maximum = torch.maximum(maximum1.float(), maximum2.float())
    weight1 = (maximum1.float() - maximum).exp() / inverse1.float()
    weight2 = (maximum2.float() - maximum).exp() / inverse2.float()
    output = torch.addcmul(first, second - first, (weight2 / (weight1 + weight2)).to(first.dtype))
    return output.transpose(1, 2).reshape(1, q.shape[-2], 4096)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('capture_root', type=Path)
    parser.add_argument('output', type=Path)
    options = parser.parse_args()
    torch.set_num_threads(32)
    torch.set_grad_enabled(False)
    kernel = compile_candidates([('prefix', dict(kind='split', mode='fast', centered=True))])['prefix']
    compiled_fused = torch.compile(fused, backend='hpu_backend', dynamic=None, options={'force_static_compile': True})
    compiled_segmented = torch.compile(segmented, backend='hpu_backend', dynamic=None, options={'force_static_compile': True})
    records = []
    for index in (7, 8):
        path = options.capture_root / f'forward-{index:02d}-layer-15-part-1.pt'
        data = torch.load(path, weights_only=True)
        prefix = data['k'].shape[1] - data['q'].shape[1]
        q = data['q'].to('hpu')
        pk, pv = (data[name][:, :prefix].contiguous().to('hpu') for name in ('k', 'v'))
        k, v = (data[name][:, prefix:].contiguous().to('hpu') for name in ('k', 'v'))
        buffer_k, buffer_v = torch.cat((pk, k), 1), torch.cat((pv, v), 1)

        def original():
            return kernel(q, torch.cat((pk, k), 1), torch.cat((pv, v), 1), 32)

        def stable():
            buffer_k[:, prefix:].copy_(k)
            buffer_v[:, prefix:].copy_(v)
            return kernel(q, buffer_k, buffer_v, 32)

        candidates = [('original', original), ('stable', stable),
                      ('fused_cat', lambda: compiled_fused(q, pk, pv, k, v)),
                      ('segmented', lambda: compiled_segmented(q, pk, pv, k, v))]
        reference = original().cpu().float()
        outputs = {name: function().cpu().float() for name, function in candidates}
        samples = {name: [] for name, _ in candidates}
        for repeat in range(5):
            for name, function in candidates if repeat % 2 == 0 else reversed(candidates):
                torch.hpu.synchronize()
                start = time.perf_counter()
                for _ in range(10):
                    function()
                torch.hpu.synchronize()
                samples[name].append((time.perf_counter() - start) / 10)
        torch.hpu.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            keys, values = torch.cat((pk, k), 1), torch.cat((pv, v), 1)
        torch.hpu.synchronize()
        cat_seconds = (time.perf_counter() - start) / 100
        row = dict(input=path.name, prefix=prefix, cat_seconds=cat_seconds,
                   stable_buffer_bytes=buffer_k.nbytes + buffer_v.nbytes,
                   candidates={name: dict(seconds=statistics.median(times), samples=times,
                       rrmse=float((outputs[name] - reference).norm() / reference.norm()),
                       finite=bool(outputs[name].isfinite().all())) for name, times in samples.items()})
        records.append(row)
        options.output.write_text(json.dumps(records, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
