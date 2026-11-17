"""Capture real Qwen DiT attention inputs without changing product execution."""
import functools
import gc
import json
import os
from pathlib import Path
import runpy
import time

import comfy.options
comfy.options.enable_args_parsing()
import execution
import torch
from comfy.ldm.qwen_image21 import model as dit
from moss_native_profile import NativeProfile

root = Path(os.environ['QWEN_D1_ROOT'])
capture = False
capture_all_layers = False
diagnostic = False
forward_index = 0
block_index = -1
attention_index = 0
records = []
forward_times = []
trajectory = None
profile_mode = ''
product_validation = os.environ.get('QWEN_D1_PRODUCT') == '1'
product_baseline = False

baseline_compiler = dit._compiled_attention
candidate_compilers = {}
if os.environ.get('QWEN_D1_CANDIDATES') and not product_validation:
    from qwen_d1_screen import candidates, compile_candidates
    names = os.environ['QWEN_D1_CANDIDATES'].split(',')
    configurations = dict(candidates())
    candidate_compilers = compile_candidates([(name, configurations[name.removesuffix('-skip0')]) for name in names])


def candidate_adapter(kernel, skip_first):
    def run(q, k, v, heads, mask, options, preferred):
        if q.shape[1] < 4096 or heads != 32 or (skip_first and options.get('block_index') == 0):
            return baseline_compiler()(q, k, v, heads, mask, options, preferred)
        return kernel(q, k, v, heads)
    return run


candidate_adapters = {name: candidate_adapter(kernel, name.endswith('-skip0')) for name, kernel in candidate_compilers.items()}

original_forward = dit.QwenImage21Transformer2DModel._forward


@functools.wraps(original_forward)
def forward(self, *args, **kwargs):
    global forward_index
    forward_index += 1
    if product_baseline:
        if len(args) > 5:
            args = list(args)
            args[5] = dict(args[5], hpu_fast_attention=False)
        else:
            kwargs['transformer_options'] = dict(kwargs.get('transformer_options', {}), hpu_fast_attention=False)
    if profile_mode and forward_index == 4:
        torch.hpu.synchronize()
        profiler = NativeProfile(root) if profile_mode == 'native' else None
        if profiler:
            profiler.__enter__()
        start = time.perf_counter()
        result = original_forward(self, *args, **kwargs)
        torch.hpu.synchronize()
        seconds = time.perf_counter() - start
        if profiler:
            profiler.__exit__(None, None, None)
        filename = 'window.json' if profiler else 'plain-window.json'
        (root / filename).write_text(json.dumps(dict(seconds=seconds, forwards=1)))
        return result
    if diagnostic:
        torch.hpu.synchronize()
    start = time.perf_counter()
    result = original_forward(self, *args, **kwargs)
    if trajectory is not None:
        trajectory.append(result.cpu())
    if diagnostic:
        torch.hpu.synchronize()
        forward_times.append(dict(index=forward_index, seconds=time.perf_counter() - start))
    return result


dit.QwenImage21Transformer2DModel._forward = forward
original_block = dit.QwenImage21TransformerBlock.forward


@functools.wraps(original_block)
def block(self, x, mod, pe, attn_fn, prefix_len, transformer_options={}):
    global block_index, attention_index
    block_index = transformer_options['block_index']
    attention_index = 0
    return original_block(self, x, mod, pe, attn_fn, prefix_len, transformer_options)


dit.QwenImage21TransformerBlock.forward = block
original_attention = dit.attention


@functools.wraps(original_attention)
def attention(q, k, v, heads, mask=None, transformer_options={}, preferred_attention=None):
    global attention_index
    attention_index += 1
    part = attention_index
    selected = capture and ((forward_index in (1, 2, 7, 8, 15, 16) and block_index in (0, 15, 31))
                            or (capture_all_layers and forward_index in (7, 8)))
    saved = None
    if selected:
        saved = dict(q=q.peek().cpu(), k=k.peek().cpu(), v=v.peek().cpu(),
                     mask=mask.cpu() if mask is not None else None, heads=heads)
    result = original_attention(q, k, v, heads, mask, transformer_options, preferred_attention)
    if selected:
        saved['output'] = result.cpu()
        suffix = '-masked' if mask is not None else ''
        name = f'forward-{forward_index:02d}-layer-{block_index:02d}-part-{part}{suffix}.pt'
        torch.save(saved, root / 'capture' / name)
        records.append(dict(file=name, forward=forward_index, layer=block_index,
                            shapes={name: list(saved[name].shape) for name in ('q', 'k', 'v')},
                            masked=mask is not None))
    return result


dit.attention = attention
original_execute = execution.PromptExecutor.execute_async


async def execute(self, prompt, prompt_id, *args, **kwargs):
    global capture, capture_all_layers, diagnostic, forward_index, trajectory, profile_mode, product_baseline
    extra = args[0] if args else kwargs.get('extra_data', {})
    capture = extra.get('d1_capture', False)
    capture_all_layers = extra.get('d1_capture_all_layers', capture)
    diagnostic = extra.get('d1_diagnostic', False)
    trajectory = [] if extra.get('d1_trajectory', False) else None
    profile_mode = extra.get('d1_profile', '')
    name = extra.get('d1_candidate', 'baseline')
    product_baseline = product_validation and name == 'baseline'
    dit._compiled_attention = baseline_compiler if product_validation or name == 'baseline' else lambda: candidate_adapters[name]
    forward_index = 0
    records.clear()
    forward_times.clear()
    gc_enabled = gc.isenabled()
    if profile_mode:
        gc.collect()
        gc.disable()
    try:
        return await original_execute(self, prompt, prompt_id, *args, **kwargs)
    finally:
        if capture or diagnostic:
            (root / (prompt_id + '-diagnostic.json')).write_text(json.dumps(
                dict(captures=records, forwards=forward_times), indent=2))
        if trajectory is not None:
            torch.save(trajectory, root / (prompt_id + '-trajectory.pt'))
            trajectory = None
        if gc_enabled:
            gc.enable()


execution.PromptExecutor.execute_async = execute


def main():
    runpy.run_path(str(Path(__file__).with_name('moss_observed_server.py')), run_name='__main__')
