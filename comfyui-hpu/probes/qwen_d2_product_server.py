"""Compare the D2 product with its D1 execution path without layer instrumentation."""
import functools
import json
import os
from pathlib import Path
import runpy
import time
import types

import comfy.options
comfy.options.enable_args_parsing()
import execution
import torch
from comfy.ldm.qwen_image21 import model as dit
from comfy.ldm.modules import attention
from moss_native_profile import NativeProfile


root = Path(os.environ['QWEN_D1_ROOT'])
baseline = False
profile_mode = ''
trajectory = None
forward_index = 0
product_fast = attention._hpu_fast_attention
product_prefix = getattr(dit.QwenImage21TransformerBlock, '_forward_with_prefix', None)
original_forward = dit.QwenImage21Transformer2DModel._forward


def baseline_fast(q, k, v):
    k = (k.float() - k.float().mean(dim=-2, keepdim=True)).to(k.dtype)
    return attention._hpu_fast_attention_part(q.contiguous(), k, v)


compilers = {}
for name in ('baseline', 'product'):
    function = types.FunctionType(dit._attention.__code__.replace(co_name='d2_product_attention_' + name), dit.__dict__)
    compilers[name] = torch.compile(function, backend='hpu_backend', dynamic=None,
                                    options={'force_static_compile': True})
dit._compiled_attention = lambda: compilers['baseline' if baseline else 'product']


if product_prefix is not None:
    def prefix(self, x, mod, pe, prefix_k, prefix_v, options):
        if baseline:
            attn_fn = dit.prefix_cached_attention(prefix_k, prefix_v, options)
            return self(x, mod, pe, attn_fn, 0, options)
        return product_prefix(self, x, mod, pe, prefix_k, prefix_v, options)

    dit.QwenImage21TransformerBlock._forward_with_prefix = prefix


@functools.wraps(original_forward)
def forward(self, *args, **kwargs):
    global forward_index
    forward_index += 1
    selected = profile_mode and forward_index == 4
    profiler = None
    if selected:
        torch.hpu.synchronize()
        if profile_mode == 'native':
            profiler = NativeProfile(root)
            profiler.__enter__()
        started = time.perf_counter()
    result = original_forward(self, *args, **kwargs)
    if selected:
        torch.hpu.synchronize()
        seconds = time.perf_counter()-started
        if profiler:
            profiler.__exit__(None, None, None)
        name = 'window.json' if profiler else 'plain-window.json'
        (root / name).write_text(json.dumps(dict(seconds=seconds, forwards=1)))
    if trajectory is not None:
        trajectory.append(result.cpu())
    return result


dit.QwenImage21Transformer2DModel._forward = forward
original_execute = execution.PromptExecutor.execute_async


async def execute(self, prompt, prompt_id, *args, **kwargs):
    global baseline, profile_mode, trajectory, forward_index
    extra = args[0] if args else kwargs.get('extra_data', {})
    baseline = extra.get('d1_candidate', 'product') == 'baseline'
    attention._hpu_fast_attention = baseline_fast if baseline else product_fast
    profile_mode = extra.get('d1_profile', '')
    trajectory = [] if extra.get('d1_trajectory', False) else None
    forward_index = 0
    try:
        return await original_execute(self, prompt, prompt_id, *args, **kwargs)
    finally:
        if trajectory is not None:
            torch.save(trajectory, root / (prompt_id + '-trajectory.pt'))
            trajectory = None


execution.PromptExecutor.execute_async = execute


def main():
    runpy.run_path(str(Path(__file__).with_name('moss_observed_server.py')), run_name='__main__')
