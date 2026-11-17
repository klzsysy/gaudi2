"""Compare attention precision and layouts on complete captured Qwen tensors."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time
import types

import torch
from comfy.cli_args import args
args.use_pytorch_cross_attention = True
from comfy.ldm.qwen_image21 import model as dit
from comfy.ldm.modules.attention import _hpu_fp8_attention_av
from habana_frameworks.torch.hpex.kernels import FusedSDPA


def attention(q, k, v, heads, kind, group=32, mode='None', recompute=False,
              precision=torch.bfloat16, centered=False, probability_scale=128.0, tile=0):
    if kind == 'baseline':
        return dit._attention(q, k, v, heads, None, {}, None)
    batch, length, width = q.shape
    q, k, v = [x.reshape(batch, x.shape[1], heads, -1).transpose(1, 2) for x in (q, k, v)]
    if centered:
        k = (k.float() - k.float().mean(dim=-2, keepdim=True)).to(k.dtype)
    if precision == torch.float16:
        q, k, v = [x.to(precision) for x in (q, k, v)]
    outputs = []
    for start in range(0, heads, group):
        query, key, value = [x[:, start:start + group] for x in (q, k, v)]
        if kind == 'split':
            split = key.shape[-2] // 128 * 128
            first, maximum1, inverse1 = torch.ops.hpu.sdpa_recomp_fwd(
                query.contiguous(), key[:, :, :split].contiguous(), value[:, :, :split].contiguous(),
                None, 0.0, query.shape[-1] ** -.5, False, True, mode, None, 'right')[:3]
            last, maximum2, inverse2 = torch.ops.hpu.sdpa_recomp_fwd(
                query.contiguous(), key[:, :, split:].contiguous(), value[:, :, split:].contiguous(),
                None, 0.0, query.shape[-1] ** -.5, False, True, mode, None, 'right')[:3]
            maximum = torch.maximum(maximum1.float(), maximum2.float())
            weight1 = (maximum1.float() - maximum).exp() / inverse1.float()
            weight2 = (maximum2.float() - maximum).exp() / inverse2.float()
            share = (weight2 / (weight1 + weight2)).to(first.dtype)
            output = torch.addcmul(first, last - first, share)
        elif kind == 'padded':
            pad = -key.shape[-2] % 128
            mask = torch.zeros((1, 1, 1, key.shape[-2] + pad), device=q.device, dtype=q.dtype)
            mask[:, :, :, key.shape[-2]:] = -torch.inf
            key = torch.nn.functional.pad(key, (0, 0, 0, pad))
            value = torch.nn.functional.pad(value, (0, 0, 0, pad))
            output = FusedSDPA.apply(query, key, value, mask, 0.0, False, None, mode, True)
        elif kind == 'fused':
            output = FusedSDPA.apply(query, key, value, None, 0.0, False, None, mode, recompute)
        elif kind == 'mixed':
            output = _hpu_fp8_attention_av(query, key, value)
        else:
            value_scale = value.abs().amax().float().clamp_min(1e-12) / 240.0
            value8 = torch.ops.hpu.cast_to_fp8_v2(value, value_scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
            query = query * query.shape[-1] ** -.25
            key = (key * key.shape[-1] ** -.25).transpose(-1, -2)
            pieces = []
            for row in range(0, length, tile or length):
                scores = query[:, :, row:row + (tile or length)] @ key
                if kind == 'explicit_fp32':
                    probabilities = scores.float().softmax(-1).to(value.dtype)
                    piece = probabilities @ value
                else:
                    probabilities = torch.ops.hpu.softmax_fp8.Scalar_scales(scores, -1, 1.0, probability_scale)
                    inverse_scale = torch.full((), 1 / probability_scale, device=q.device, dtype=torch.float32)
                    piece = torch.ops.hpu.fp8_gemm_v2(probabilities, False, value8, False, None,
                                                    q.dtype, inverse_scale, value_scale, None, False)
                pieces.append(piece)
            output = torch.cat(pieces, dim=-2)
        outputs.append(output)
    return torch.cat(outputs, dim=1).transpose(1, 2).reshape(batch, length, width).to(torch.bfloat16)


def candidates():
    values = [('baseline', dict(kind='baseline')), ('mixed', dict(kind='mixed', group=1))]
    for mode in ('None', 'fast', 'fp32'):
        for group in (32, 8, 1):
            for recompute in (False, True):
                values.append((f'fused-{mode}-g{group}-r{int(recompute)}',
                               dict(kind='fused', mode=mode, group=group, recompute=recompute)))
    for mode in ('None', 'fast'):
        values.append((f'fp16-{mode}', dict(kind='fused', mode=mode, precision=torch.float16)))
        values.append((f'split-{mode}', dict(kind='split', mode=mode)))
        values.append((f'padded-{mode}', dict(kind='padded', mode=mode)))
    values.append(('center-fast', dict(kind='fused', mode='fast', centered=True)))
    values.append(('center-split-fast', dict(kind='split', mode='fast', centered=True)))
    values.append(('explicit-fp32', dict(kind='explicit_fp32', group=1)))
    for scale in (32., 128., 512., 2048.):
        values.append((f'mixed-scale{int(scale)}', dict(kind='scaled', group=1, probability_scale=scale)))
    for tile in (1024, 2048):
        values.append((f'mixed-tile{tile}', dict(kind='scaled', group=1, tile=tile)))
    return values


def compile_candidates(selected):
    compiled = {}
    for name, config in selected:
        defaults = (config['kind'], config.get('group', 32), config.get('mode', 'None'),
                    config.get('recompute', False), config.get('precision', torch.bfloat16),
                    config.get('centered', False), config.get('probability_scale', 128.0), config.get('tile', 0))
        function = types.FunctionType(attention.__code__.replace(co_name='d1_' + name), globals(), argdefs=defaults)
        compiled[name] = torch.compile(function, backend='hpu_backend', dynamic=None,
                                       options={'force_static_compile': True})
    return compiled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--all-inputs', action='store_true')
    parser.add_argument('--names', nargs='+')
    parser.add_argument('--output', default='screen')
    options = parser.parse_args()
    torch.set_num_threads(32)
    torch.set_grad_enabled(False)
    inputs = sorted((options.root / 'capture').glob('*.pt'))
    rows = []
    selected = candidates()
    if options.names:
        selected = [(name, config) for name, config in selected if name in options.names]
    compiled = compile_candidates(selected)
    for path in inputs:
        saved = torch.load(path, weights_only=False)
        if saved['mask'] is not None:
            continue
        if not options.all_inputs and path.name not in ('forward-07-layer-15-part-1.pt', 'forward-08-layer-15-part-1.pt'):
            continue
        heads = saved['heads']
        q, k, v = [saved[name].to('hpu') for name in ('q', 'k', 'v')]
        sample_rows = torch.linspace(0, q.shape[1] - 1, 17).long()
        cpu_q, cpu_k, cpu_v = [saved[name].reshape(1, -1, heads, 128)[0].transpose(0, 1).double() for name in ('q', 'k', 'v')]
        scores = cpu_q[:, sample_rows] @ cpu_k.transpose(-1, -2) / 128 ** .5
        accurate = scores.softmax(-1) @ cpu_v
        reference = saved['output'].float()
        for name, _ in selected:
            gc.collect()
            try:
                started = time.perf_counter()
                output = compiled[name](q, k, v, heads)
                torch.hpu.synchronize()
                cold = time.perf_counter() - started
                actual = output.cpu().float()
                sample = actual.reshape(1, -1, heads, 128)[0].transpose(0, 1)[:, sample_rows].double()
                errors = (sample - accurate).square().sum((1, 2)).sqrt() / accurate.square().sum((1, 2)).sqrt()
                times = []
                for _ in range(5):
                    torch.hpu.synchronize()
                    start = time.perf_counter()
                    for _ in range(3):
                        output = compiled[name](q, k, v, heads)
                    torch.hpu.synchronize()
                    times.append((time.perf_counter() - start) / 3)
                row = dict(input=path.name, name=name, cold=cold, seconds=statistics.median(times), times=times,
                           rrmse=float((actual - reference).norm() / reference.norm()),
                           fp64_rrmse=float((sample - accurate).norm() / accurate.norm()),
                           head_rrmse=errors.tolist(), finite=bool(actual.isfinite().all()),
                           max_input={key: float(saved[key].abs().max()) for key in ('q', 'k', 'v')})
            except (RuntimeError, AssertionError) as error:
                row = dict(input=path.name, name=name, error=str(error))
            rows.append(row)
            (options.root / (options.output + '.json')).write_text(json.dumps(rows, indent=2))
            print(json.dumps({key: value for key, value in row.items() if key not in ('head_rrmse', 'times')}), flush=True)


if __name__ == '__main__':
    main()
