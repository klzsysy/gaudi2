"""Validate changing inputs and measure complete D2 native attention calls."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from comfy.cli_args import args as comfy_args
comfy_args.use_pytorch_cross_attention = True
from comfy.ldm.modules.attention import _hpu_fast_attention
from qwen_d2_native_runtime import NativeAttention


def framework_check(matrix, other, vector):
    scale = torch.full((), .125, dtype=torch.float32, device=matrix.device)
    left = torch.ops.hpu.cast_to_fp8_v2(matrix, 8.0, False, False, torch.float8_e4m3fn)[0]
    right = torch.ops.hpu.cast_to_fp8_v2(other, 8.0, False, False, torch.float8_e4m3fn)[0]
    product = torch.ops.hpu.fp8_gemm_v2(left, False, right, False, None, torch.bfloat16,
                                      scale, scale, None, False)
    return (matrix @ other, product, vector.exp(), vector.sigmoid(), vector.tanh(),
            vector.rsqrt(), vector.log(), vector.sin(), vector.cos(), vector.softmax(-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--all', action='store_true')
    options = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(32)
    baseline = torch.compile(_hpu_fast_attention, backend='hpu_backend', dynamic=None,
                             options={'force_static_compile': True})
    generator = torch.Generator().manual_seed(814)
    checks = (torch.randn(128, 128, generator=generator).bfloat16().to('hpu'),
              torch.randn(128, 256, generator=generator).bfloat16().to('hpu'),
              (torch.rand(64, 128, generator=generator) * 3 + .25).to('hpu'))
    framework = torch.compile(framework_check, backend='hpu_backend', dynamic=False)
    expected_checks = [value.cpu() for value in framework(*checks)]
    runtime = NativeAttention(options.root)
    results = []
    try:
        paths = sorted(options.inputs.glob('*.pt'))
        for path in paths:
            if 'masked' in path.name:
                continue
            if not options.all and path.name not in ('forward-07-layer-15-part-1.pt', 'forward-08-layer-15-part-1.pt'):
                continue
            saved = torch.load(path, weights_only=True)
            tensors = [saved[name].reshape(1, -1, 32, 128).transpose(1, 2).contiguous() for name in ('q', 'k', 'v')]
            q, k, v = [tensor.to('hpu') for tensor in tensors]
            actual = runtime(q, k, v).cpu()
            rows = torch.linspace(0, q.shape[-2]-1, 17).long()
            cpu_q, cpu_k, cpu_v = [tensor.double() for tensor in tensors]
            reference = ((cpu_q[:, :, rows] @ cpu_k.transpose(-1, -2))/128**.5).softmax(-1) @ cpu_v
            sample = actual[:, :, rows].double()
            errors = (sample-reference).square().sum((0, 2, 3)).sqrt()/reference.square().sum((0, 2, 3)).sqrt()
            assert actual.isfinite().all()
            record = dict(input=path.name, fp64_rrmse=float((sample-reference).norm()/reference.norm()),
                          head_rrmse=errors.tolist(), call=runtime.records[-1])
            if path.name in ('forward-07-layer-15-part-1.pt', 'forward-08-layer-15-part-1.pt'):
                first = actual.clone()
                altered = runtime(q, k, v + .25).cpu()
                returned = runtime(q, k, v).cpu()
                assert torch.equal(first, returned)
                assert not torch.equal(first, altered)
                query_changed = runtime(q.roll(32, dims=-2), k, v).cpu()
                key_changed = runtime(q, k.roll(64, dims=-2), v).cpu()
                assert not torch.equal(first, query_changed)
                assert not torch.equal(first, key_changed)
                assert torch.equal(first, runtime(q, k, v).cpu())
                assert torch.equal(q.cpu(), tensors[0]) and torch.equal(k.cpu(), tensors[1]) and torch.equal(v.cpu(), tensors[2])
                baseline(q, k, v)
                torch.hpu.synchronize()
                times = {'baseline': [], 'native': []}
                for index in range(7):
                    order = [('baseline', baseline), ('native', runtime)]
                    if index % 2:
                        order.reverse()
                    for name, function in order:
                        torch.hpu.synchronize()
                        started = time.perf_counter()
                        for _ in range(3):
                            output = function(q, k, v)
                        torch.hpu.synchronize()
                        times[name].append((time.perf_counter()-started)/3)
                record.update(value_change=True, query_change=True, key_change=True,
                              return_equal=True, inputs_unchanged=True, times=times,
                              seconds={name: statistics.median(values) for name, values in times.items()})
            results.append(record)
            (options.output / 'validation.json').write_text(json.dumps(results, indent=2))
            print(json.dumps({key: value for key, value in record.items() if key not in ('head_rrmse', 'times')}), flush=True)
    finally:
        runtime.close()
    check_equal = [torch.equal(value.cpu(), expected) for value, expected in zip(framework(*checks), expected_checks)]
    (options.output / 'framework-checks.json').write_text(json.dumps(dict(
        names=['bf16_mm', 'fp8_mm', 'exp', 'sigmoid', 'tanh', 'rsqrt', 'log', 'sin', 'cos', 'softmax'],
        bitwise_equal=check_equal), indent=2))
    assert all(check_equal), check_equal
    (options.output / 'calls.json').write_text(json.dumps(runtime.records, indent=2))


if __name__ == '__main__':
    main()
