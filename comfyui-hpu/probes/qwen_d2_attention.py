"""Export and time D2 attention subgraphs on recorded Qwen inputs."""
import argparse
import ctypes
import json
from pathlib import Path
import statistics
import time

import torch
from comfy.cli_args import args as comfy_args
comfy_args.use_pytorch_cross_attention = True
from comfy.ldm.modules.attention import _hpu_fast_attention
from torch._dynamo.backends.registry import lookup_backend
from torch._higher_order_ops.hints_wrap import hints_wrapper


def explicit_tile(q, k, v, value_scale, mixed):
    scores = q @ k.transpose(-1, -2)
    if mixed:
        probabilities = torch.ops.hpu.softmax_fp8.Scalar_scales(scores, -1, 1.0, 128.0)
        inverse_scale = torch.full((), 1 / 128.0, device=q.device, dtype=torch.float32)
        output = torch.ops.hpu.fp8_gemm_v2(probabilities, False, v, False, None,
                                         q.dtype, inverse_scale, value_scale, None, False)
    else:
        output = scores.softmax(-1) @ v
    return (output,)


def tiled_attention(q, k, v, tile, mode, mixed):
    k = (k.float() - k.float().mean(dim=-2, keepdim=True)).to(k.dtype)
    q, k, v = [tensor.contiguous() for tensor in (q, k, v)]
    if mode != 'loop':
        q, k = q * q.shape[-1] ** -.25, k * k.shape[-1] ** -.25
    value_scale = v.abs().amax().float().clamp_min(1e-12) / 240.0
    if mixed:
        v = torch.ops.hpu.cast_to_fp8_v2(v, value_scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
    outputs = []
    for start in range(0, q.shape[-2], tile):
        query = q[:, :, start:start + tile].contiguous()
        if mode == 'loop':
            split = k.shape[-2] // 128 * 128
            first, maximum1, inverse1 = torch.ops.hpu.sdpa_recomp_fwd(
                query, k[:, :, :split].contiguous(), v[:, :, :split].contiguous(),
                None, 0.0, q.shape[-1] ** -.5, False, True, 'fast', None, 'right')[:3]
            last, maximum2, inverse2 = torch.ops.hpu.sdpa_recomp_fwd(
                query, k[:, :, split:].contiguous(), v[:, :, split:].contiguous(),
                None, 0.0, q.shape[-1] ** -.5, False, True, 'fast', None, 'right')[:3]
            maximum = torch.maximum(maximum1.float(), maximum2.float())
            weight1 = (maximum1.float() - maximum).exp() / inverse1.float()
            weight2 = (maximum2.float() - maximum).exp() / inverse2.float()
            share = (weight2 / (weight1 + weight2)).to(first.dtype)
            output = torch.addcmul(first, last - first, share)
        elif mode == 'hinted':
            output = hints_wrapper(explicit_tile, (query, k, v, value_scale, mixed), {},
                                   hints={'user_bundle_id': start // tile})[0]
        else:
            output = explicit_tile(query, k, v, value_scale, mixed)[0]
        outputs.append(output)
    return torch.cat(outputs, dim=-2)


def folded_attention(q, k, v, tile):
    k = (k.float() - k.float().mean(dim=-2, keepdim=True)).to(k.dtype)
    batch, heads, length, width = q.shape
    padded = (length + tile - 1) // tile * tile
    query = torch.nn.functional.pad(q, (0, 0, 0, padded - length))
    query = query.reshape(batch * heads, padded // tile, tile, width).contiguous()
    key = k.reshape(batch * heads, 1, k.shape[-2], width)
    value = v.contiguous().reshape(batch * heads, 1, v.shape[-2], width)
    split = key.shape[-2] // 128 * 128
    scale = width ** -.5
    first, maximum1, inverse1 = torch.ops.hpu.sdpa_recomp_fwd(
        query, key[:, :, :split].contiguous(), value[:, :, :split].contiguous(),
        None, 0.0, scale, False, True, 'fast', None, 'right')[:3]
    last, maximum2, inverse2 = torch.ops.hpu.sdpa_recomp_fwd(
        query, key[:, :, split:].contiguous(), value[:, :, split:].contiguous(),
        None, 0.0, scale, False, True, 'fast', None, 'right')[:3]
    maximum = torch.maximum(maximum1.float(), maximum2.float())
    weight1 = (maximum1.float() - maximum).exp() / inverse1.float()
    weight2 = (maximum2.float() - maximum).exp() / inverse2.float()
    share = (weight2 / (weight1 + weight2)).to(first.dtype)
    output = torch.addcmul(first, last - first, share)
    return output.reshape(batch, heads, padded, width)[:, :, :length]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--mode', choices=('baseline', 'folded', 'loop', 'explicit', 'hinted'), default='baseline')
    parser.add_argument('--tile', type=int, default=256)
    parser.add_argument('--heads', type=int, default=32)
    parser.add_argument('--query-rows', type=int)
    parser.add_argument('--mixed', action='store_true')
    parser.add_argument('--configuration', default='{}')
    parser.add_argument('--native', action='store_true')
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(32)
    saved = torch.load(options.input, weights_only=False)
    tensors = [saved[name].reshape(1, -1, saved['heads'], 128).transpose(1, 2)[:, :options.heads]
               for name in ('q', 'k', 'v')]
    if options.query_rows:
        tensors[0] = tensors[0][:, :, :options.query_rows]
    q, k, v = [tensor.to('hpu') for tensor in tensors]
    api = ctypes.CDLL('/usr/lib/habanalabs/libSynapse.so')
    api.synConfigurationSet.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    api.synConfigurationGet.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint64]
    requested = json.loads(options.configuration)
    if requested:
        for name, value in dict(ENABLE_EXPERIMENTAL_FLAGS='1', **requested).items():
            assert api.synConfigurationSet(name.encode(), str(value).encode()) == 0
    configuration = {}
    names = ('ENABLE_LAYERED_PIPELINE_BRAIN', 'ENABLE_LAYERED_BRAIN_ATTENTION_SEEDS',
             'ENABLE_LAYERED_BRAIN_MULTI_MME_ATTENTION_SEEDS', 'ENABLE_FLASH_ATTENTION_SLICING',
             'SRAM_SLICER_MAX_CAPACITY_BYTES', 'DEFAULT_PIPELINE_DEPTH')
    for name in dict.fromkeys((*names, *requested)):
        buffer = ctypes.create_string_buffer(1024)
        status = api.synConfigurationGet(name.encode(), buffer, 1024)
        configuration[name] = dict(status=status, value=buffer.value.decode())
    (options.output / 'configuration.json').write_text(json.dumps(configuration, indent=2))
    assert all(configuration[name]['value'] == str(value) for name, value in requested.items())
    backend = lookup_backend('hpu_backend')
    graphs = []

    def capture_backend(graph, inputs, **kwargs):
        graphs.append(dict(code=graph.code, nodes=[dict(name=node.name, target=str(node.target))
                                                  for node in graph.graph.nodes]))
        (options.output / 'fx.json').write_text(json.dumps(graphs, indent=2))
        return backend(graph, inputs, **kwargs)

    if options.mode == 'baseline':
        function, arguments = _hpu_fast_attention, (q, k, v)
    elif options.mode == 'folded':
        function, arguments = folded_attention, (q, k, v, options.tile)
    else:
        function, arguments = tiled_attention, (q, k, v, options.tile, options.mode, options.mixed)
    compiled = torch.compile(function, backend=capture_backend, dynamic=False,
                             options={'force_static_compile': True})
    started = time.perf_counter()
    output = compiled(*arguments)
    torch.hpu.synchronize()
    cold = time.perf_counter() - started
    actual = output.cpu()
    assert actual.isfinite().all()
    for _ in range(4):
        output = compiled(*arguments)
    torch.hpu.synchronize()
    times = []
    for _ in range(7):
        started = time.perf_counter()
        for _ in range(10):
            output = compiled(*arguments)
        torch.hpu.synchronize()
        times.append((time.perf_counter() - started) / 10)
    assert torch.equal(actual, output.cpu())
    sample_rows = torch.linspace(0, q.shape[-2] - 1, 17).long()
    cpu_q, cpu_k, cpu_v = [tensor.double() for tensor in tensors]
    reference = ((cpu_q[:, :, sample_rows] @ cpu_k.transpose(-1, -2)) / 128 ** .5).softmax(-1) @ cpu_v
    sample = actual[:, :, sample_rows].double()
    errors = (sample - reference).square().sum((0, 2, 3)).sqrt() / reference.square().sum((0, 2, 3)).sqrt()
    result = dict(input=str(options.input), mode=options.mode, heads=options.heads, tile=options.tile,
                  shape=[list(tensor.shape) for tensor in tensors], cold=cold, times=times,
                  seconds=statistics.median(times), fp64_rrmse=float((sample-reference).norm()/reference.norm()),
                  head_rrmse=errors.tolist(), finite=True, repeat_equal=True, graphs=len(graphs))
    if options.native:
        from moss_native_profile import NativeProfile
        with NativeProfile(options.output):
            started = time.perf_counter()
            for _ in range(10):
                output = compiled(*arguments)
            torch.hpu.synchronize()
            result['profile_seconds'] = (time.perf_counter() - started) / 10
        result['profile_perturbation'] = result['profile_seconds'] / result['seconds'] - 1
    torch.save(actual, options.output / 'output.pt')
    (options.output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
