"""Replay captured real DiT block tensors against existing HPU backends."""
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True
import comfy.ops
import comfy.ldm.moss_image.model as moss
if hasattr(moss, '_attention'):
    moss.attention = moss._attention
import habana_frameworks.torch.utils.debug as debug
from habana_frameworks.torch.hpex.kernels import FusedSDPA

torch.set_grad_enabled(False)
torch.set_num_threads(16)
directory = Path(os.environ['MOSS_DIT_ROOT'])
output = directory / os.environ.get('MOSS_REPLAY_NAME', 'replay')
output.mkdir(exist_ok=True)
source = torch.load(directory / 'original/block-input-0.pt', weights_only=True)
state = source['state']
dim = state['norm1.weight'].numel()
ffn = state['proj.ffn_up_gate.weight'].shape[0] // 2
block = moss.Block(dim, ffn, source['heads'], fused_qkv=source['fused_qkv'], dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
block.load_state_dict(state)
block = block.to('hpu')


def hpu(value):
    if isinstance(value, torch.Tensor):
        return value.to('hpu')
    if isinstance(value, (tuple, list)):
        return type(value)(hpu(v) for v in value)
    return value


x, mod, pe = hpu(source['args'])
mask = hpu(source['mask'])
work = x.clone()
results = {'shape': list(x.shape), 'heads': source['heads']}


def measure(fn, iterations=10):
    start = time.perf_counter()
    value = fn()
    torch.hpu.synchronize()
    cold = time.perf_counter() - start
    for _ in range(3):
        value = fn()
    torch.hpu.synchronize()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(iterations):
            value = fn()
        torch.hpu.synchronize()
        samples.append((time.perf_counter() - start) / iterations)
    return value, {'cold_seconds': cold, 'median_seconds': statistics.median(samples), 'samples': samples}


def error(value, reference):
    value, reference = value.float().cpu(), reference.float().cpu()
    delta = value - reference
    return {'finite': bool(torch.isfinite(value).all()), 'max_abs': float(delta.abs().max()),
            'mean_abs': float(delta.abs().mean()), 'relative_rmse': float(delta.square().mean().sqrt() / reference.square().mean().sqrt()),
            'cosine': float(torch.nn.functional.cosine_similarity(value.flatten().double(), reference.flatten().double(), dim=0))}


def run(fn=block):
    work.copy_(x)
    return fn(work, mod, pe, {}, mask=mask)


baseline, results['original'] = measure(run)
baseline = baseline.cpu().clone()
torch.save(baseline, output / 'original-output.pt')
original_sdpa = comfy.ops.scaled_dot_product_attention
captured = []
def capture_sdpa(q, k, v, *values, **kwargs):
    captured.append((q.clone(), k.clone(), v.clone(), values, kwargs))
    return original_sdpa(q, k, v, *values, **kwargs)
comfy.ops.scaled_dot_product_attention = capture_sdpa
run()
torch.hpu.synchronize()
comfy.ops.scaled_dot_product_attention = original_sdpa
q, k, v, values, kwargs = captured[0]
print('CAPTURE', q.shape, q.stride(), k.stride(), v.stride(), kwargs, flush=True)

attn_ref, results['attention-original'] = measure(lambda: original_sdpa(q, k, v, *values, **kwargs))
attn_ref = attn_ref.cpu()
for name, recompute, softmax in [('sdpa', False, 'None'), ('sdpa-recompute', True, 'None'), ('sdpa-fp32', False, 'fp32'), ('sdpa-fp32-recompute', True, 'fp32')]:
    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        return FusedSDPA.apply(q, k, v, attn_mask, dropout_p, is_causal, scale, softmax, recompute)
    try:
        actual, result = measure(lambda: sdpa(q, k, v, *values, **kwargs))
        result['error'] = error(actual, attn_ref)
        results['attention-' + name] = result
        comfy.ops.scaled_dot_product_attention = sdpa
        actual, result = measure(run)
        result['error'] = error(actual, baseline)
        results[name] = result
    except Exception:
        results[name] = {'exception': traceback.format_exc()}
    finally:
        comfy.ops.scaled_dot_product_attention = original_sdpa
    print(name, results[name], flush=True)
    (output / 'results.json').write_text(json.dumps(results, indent=2))

for name, function in [('block-compile', block), ('attention-compile', original_sdpa)]:
    try:
        compiled = torch.compile(function, backend='hpu_backend', fullgraph=True, dynamic=False)
        if name == 'block-compile':
            actual, result = measure(lambda: run(compiled))
            reference = baseline
        else:
            actual, result = measure(lambda: compiled(q, k, v, *values, **kwargs))
            reference = attn_ref
        result['error'] = error(actual, reference)
        results[name] = result
    except Exception:
        results[name] = {'exception': traceback.format_exc()}
    print(name, results[name], flush=True)

if os.environ.get('MOSS_DIT_TRACE', '1') == '1':
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU], record_shapes=True) as prof:
        run()
        torch.hpu.synchronize()
    prof.profiler.export_chrome_trace(str(output / 'original.trace.json'))
    (output / 'cpu-ops.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=45))

results['fallback'] = debug._get_fallback_op_count()
(output / 'results.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2), flush=True)

if os.environ.get('MOSS_DIT_EXTENDED') == '1':
    def fp16_linear_wanted(x):
        return x.is_cuda and x.dtype == torch.float16 and getattr(torch.backends.cuda.matmul, 'allow_fp16_accumulation', False)
    comfy.ops._fp16_linear_wanted = fp16_linear_wanted
    original_attention = moss.attention
    for name in ['attention-region', 'block-fixed', 'block-decomposed']:
        torch._dynamo.reset()
        if name == 'attention-region':
            moss.attention = torch.compile(original_attention, backend='hpu_backend', fullgraph=True, dynamic=False)
            function = block
        else:
            moss.attention = original_attention
            if name == 'block-decomposed':
                from habana_frameworks.torch.dynamo.compile_backend.decomposition import hpu_backend_decompositions_common
                from comfy_kitchen.backends.eager.adaln import rms_adaln
                from comfy_kitchen.backends.eager.rope import rms_rope_split_half
                hpu_backend_decompositions_common[torch.ops.comfy_kitchen.rms_adaln.default] = rms_adaln
                hpu_backend_decompositions_common[torch.ops.comfy_kitchen.rms_rope_split_half.default] = rms_rope_split_half
            function = torch.compile(block, backend='hpu_backend', fullgraph=True, dynamic=False)
        try:
            actual, result = measure(lambda: run(function))
            result['error'] = error(actual, baseline)
            results[name] = result
            if os.environ.get('MOSS_DIT_TRACE', '1') == '1':
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU], record_shapes=True) as prof:
                    run(function)
                    torch.hpu.synchronize()
                prof.profiler.export_chrome_trace(str(output / (name + '.trace.json')))
        except Exception:
            results[name] = {'exception': traceback.format_exc()}
        print(name, results[name], flush=True)
        (output / 'results.json').write_text(json.dumps(results, indent=2))
