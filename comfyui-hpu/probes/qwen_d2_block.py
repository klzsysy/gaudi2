"""Measure a wider compiled region on real D1 FP8 transformer blocks."""
import argparse
import ctypes
import gc
import json
from pathlib import Path
import statistics
import time

import torch
from comfy.cli_args import args as comfy_args
comfy_args.disable_dynamic_vram = True
comfy_args.use_pytorch_cross_attention = True
import comfy.sd
import comfy.model_management
from comfy.ldm.qwen_image21 import model as dit
from torch._dynamo.backends.registry import lookup_backend
from comfy_kitchen.backends.eager import adaln as eager_adaln

original_modulated_norm = dit._modulated_norm


def inlined_modulated_norm(norm, x, scale, prefix_len, zero):
    if prefix_len == 0:
        return eager_adaln(x, scale[1], zero, norm.eps)
    return original_modulated_norm(norm, x, scale, prefix_len, zero)


def forward(block, x, mod, pe, prefix_k, prefix_v, fast):
    settings = dict(hpu_fast_attention=fast, block_index=1 if fast else 0)
    attn_fn = dit.prefix_cached_attention(prefix_k, prefix_v, settings)
    return block(x, mod, pe, attn_fn, 0, settings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--layer', type=int, default=15)
    parser.add_argument('--dump', action='store_true')
    parser.add_argument('--inline-norm', action='store_true')
    options = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(32)
    patcher = comfy.sd.load_diffusion_model(
        '/home/sonny/ComfyUI-Models/models/diffusion_models/qwen_image_2.1_turbo_fp8.safetensors')
    comfy.model_management.load_models_gpu([patcher])
    torch.hpu.synchronize()
    if options.dump:
        api = ctypes.CDLL('/usr/lib/habanalabs/libSynapse.so')
        api.synConfigurationSet.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        assert api.synConfigurationSet(b'DUMP_POST_GRAPHS', str(options.output / 'post').encode()) == 0
    backend = lookup_backend('hpu_backend')
    graphs = []

    def capture_backend(graph, inputs, **kwargs):
        graphs.append(graph.code)
        (options.output / 'fx.json').write_text(json.dumps(graphs, indent=2))
        return backend(graph, inputs, **kwargs)

    compiled = torch.compile(forward, backend=capture_backend, dynamic=None,
                             options={'force_static_compile': True})
    results = []
    for path in sorted(options.inputs.glob(f'*-layer-{options.layer:02d}.pt')):
        saved = torch.load(path, weights_only=True)
        block = patcher.model.diffusion_model.transformer_blocks[saved['layer']]
        x, pe, prefix_k, prefix_v = [saved[name].to('hpu') for name in ('x', 'pe', 'prefix_k', 'prefix_v')]
        mod = tuple(tuple(t.to('hpu') for t in item) if isinstance(item, tuple) else item.to('hpu') for item in saved['mod'])
        outputs, times, cold = {}, {'baseline': [], 'compiled': []}, {}
        for name, function in [('baseline', forward), ('compiled', compiled)]:
            dit._modulated_norm = inlined_modulated_norm if options.inline_norm and name == 'compiled' else original_modulated_norm
            started = time.perf_counter()
            output = function(block, x.clone(), mod, pe, prefix_k, prefix_v, saved['layer'] != 0)
            torch.hpu.synchronize()
            cold[name] = time.perf_counter()-started
            outputs[name] = output.cpu()
            assert outputs[name].isfinite().all()
            for _ in range(3):
                function(block, x.clone(), mod, pe, prefix_k, prefix_v, saved['layer'] != 0)
            torch.hpu.synchronize()
        for index in range(7):
            order = [('baseline', forward), ('compiled', compiled)]
            if index % 2:
                order.reverse()
            for name, function in order:
                dit._modulated_norm = inlined_modulated_norm if options.inline_norm and name == 'compiled' else original_modulated_norm
                gc.collect()
                torch.hpu.synchronize()
                started = time.perf_counter()
                for _ in range(5):
                    output = function(block, x.clone(), mod, pe, prefix_k, prefix_v, saved['layer'] != 0)
                torch.hpu.synchronize()
                times[name].append((time.perf_counter()-started)/5)
                assert torch.equal(output.cpu(), outputs[name]), (path.name, name, 'repeat')
        delta = outputs['compiled'].float()-outputs['baseline'].float()
        record = dict(input=path.name, cold=cold, times=times,
                      seconds={name: statistics.median(values) for name, values in times.items()},
                      baseline_matches_capture=torch.equal(outputs['baseline'], saved['output']),
                      bitwise_equal=torch.equal(outputs['compiled'], outputs['baseline']),
                      rrmse=float(delta.norm()/outputs['baseline'].float().norm()),
                      max_abs=float(delta.abs().max()), graph_count=len(graphs))
        results.append(record)
        (options.output / 'block-results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
