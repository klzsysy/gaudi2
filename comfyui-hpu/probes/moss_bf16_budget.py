"""Real-input BF16 matrix/layout and block budget; exclusive HPU process."""
import json
import os
from pathlib import Path
import statistics
import time

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True
import comfy.ops
from comfy.ldm.moss_image import model as moss
import habana_frameworks.torch.utils.debug as debug

torch.set_grad_enabled(False)
torch.set_num_threads(16)
root = Path(os.environ['MOSS_BF16_ROOT'])
output = root / 'budget.json'
results = []
state = torch.load(root / 'captures/weights-0.pt', weights_only=True)
block = moss.Block(3072, 8192, 24, fused_qkv=True, dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
block.load_state_dict(state)
block.to('hpu')


def compiled(fn):
    return torch.compile(fn, backend='hpu_backend', dynamic=None, options={'force_static_compile': True})


def measure(fn, iterations=20):
    start = time.perf_counter()
    first = fn()
    torch.hpu.synchronize()
    cold = time.perf_counter() - start
    reference = first.detach().cpu().float()
    del first
    for _ in range(3):
        fn()
    torch.hpu.synchronize()
    times = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.hpu.synchronize()
        times.append((time.perf_counter() - start) / iterations)
    return reference, {'cold_s': cold, 'median_ms': statistics.median(times) * 1000,
                       'samples_ms': [v * 1000 for v in times]}


def record(label, fn, condition, reference=None, flops=None, metadata=None, iterations=20):
    value, timing = measure(fn, iterations)
    entry = {'condition': condition, 'label': label, **timing, **(metadata or {})}
    if reference is not None:
        delta = value - reference
        entry['error'] = {'rrmse': float(delta.square().mean().sqrt() / reference.square().mean().sqrt()),
                          'max_abs': float(delta.abs().max()), 'finite': bool(value.isfinite().all())}
    if flops is not None:
        entry['tflops'] = flops / (entry['median_ms'] * 1e9)
    results.append(entry)
    output.write_text(json.dumps({'results': results, 'fallback': debug._get_fallback_op_count()}, indent=2))
    print(json.dumps(entry), flush=True)
    return value


def linear(x, weight):
    return torch.nn.functional.linear(x, weight)


def pretransposed(x, transposed_weight):
    return torch.matmul(x, transposed_weight)


def module_linear(module, x):
    return module(x)


linear_compiled = compiled(linear)
module_compiled = compiled(module_linear)
transposed_compiled = compiled(pretransposed)
for condition in (307, 23):
    data = torch.load(root / f'captures/input-{condition}-0-0.pt', weights_only=True)
    x = data['args'][0].to('hpu')
    mod = tuple(v.to('hpu') for v in data['args'][1])
    pe = data['args'][2].to('hpu')
    y1 = moss.modulate(x, block.norm1, mod, 0, None)
    attention_fn, ffn_fn = moss._compiled_block_ops()
    a = attention_fn(block, y1, pe, {})
    residual = moss.residual(x.clone(), a, mod, 2, None)
    y2 = moss.modulate(residual, block.norm2, mod, 3, None)
    up_gate = block.proj['ffn_up_gate'](y2)
    activated = comfy.ops._eager_input_act(up_gate, 'situ')
    values = [('qkv', y1), ('o', a), ('ffn_up_gate', y2), ('ffn_down', activated)]
    # Capture the actual pre-output-projection activation for its layout and data.
    original = moss._attention
    captured = []
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result
    moss._attention = capture
    block._forward_attention(y1, pe, {})
    moss._attention = original
    values[1] = ('o', captured[0])
    for name, value in values:
        weight = block.proj[name].weight
        flops = 2 * value.numel() * weight.shape[0]
        info = {'input_shape': list(value.shape), 'input_stride': list(value.stride()),
                'weight_shape': list(weight.shape), 'weight_stride': list(weight.stride())}
        ref = record(name + '/linear', lambda: linear(value, weight), condition, flops=flops, metadata=info)
        record(name + '/linear-compiled', lambda: linear_compiled(value, weight), condition, ref, flops)
        record(name + '/module', lambda: block.proj[name](value), condition, ref, flops)
        record(name + '/module-compiled', lambda: module_compiled(block.proj[name], value), condition, ref, flops)
        wt = weight.t().contiguous()
        record(name + '/transposed-compiled', lambda: transposed_compiled(value, wt), condition, ref, flops,
               metadata={'extra_weight_bytes': wt.numel() * wt.element_size()})
        del wt, ref
    record('modulation', lambda: moss.modulate(x, block.norm1, mod, 0, None), condition)
    record('attention-region', lambda: attention_fn(block, y1, pe, {}), condition)
    record('ffn-region', lambda: ffn_fn(block, y2), condition)
    record('residual-with-copy', lambda: moss.residual(x.clone(), a, mod, 2, None), condition)
    record('block-with-copy', lambda: block(x.clone(), mod, pe, {}), condition, iterations=10)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU], record_shapes=True) as profile:
        with torch.profiler.record_function('real_block'):
            block(x.clone(), mod, pe, {})
        torch.hpu.synchronize()
    profile.profiler.export_chrome_trace(str(root / f'block-{condition}.trace.json'))
    (root / f'block-{condition}.cpu.txt').write_text(profile.key_averages().table(sort_by='self_cpu_time_total', row_limit=50))
