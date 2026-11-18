"""Capture complete D1 block inputs for D2 dataflow experiments."""
import functools
import inspect
import json
import os
from pathlib import Path
import types

import qwen_d1_server as observed
import torch
from comfy.ldm.qwen_image21 import model as dit
from comfy.ldm.modules import attention as attention_module
from qwen_d2_attention import tiled_attention
from qwen_d2_native_runtime import NativeAttention
from comfy_kitchen.backends.eager import adaln as eager_adaln


root = Path(os.environ['QWEN_D1_ROOT'])
original_block = dit.QwenImage21TransformerBlock.forward
mode = 'baseline'
original_fast = attention_module._hpu_fast_attention
original_modulated_norm = dit._modulated_norm
native_runtime = None
native_fallback = torch.compile(original_fast, backend='hpu_backend', dynamic=None,
                                 options={'force_static_compile': True})


@torch.compiler.disable
def native_attention(q, k, v):
    global native_runtime
    if q.shape != (1, 32, 8160, 128) or k.shape[-2] > 9472:
        return native_fallback(q, k, v)
    if native_runtime is None:
        native_runtime = NativeAttention(os.environ['QWEN_D2_NATIVE_ROOT'], valid=k.shape[-2])
    return native_runtime(q, k, v)


def inlined_modulated_norm(norm, x, scale, prefix_len, zero):
    if prefix_len == 0:
        return eager_adaln(x, scale[1], zero, norm.eps)
    return original_modulated_norm(norm, x, scale, prefix_len, zero)


def loop_attention(q, k, v):
    return tiled_attention(q, k, v, 2048, 'loop', False)


def prefix_block(self, x, mod, pe, prefix_k, prefix_v, fast):
    settings = dict(hpu_fast_attention=fast, block_index=1 if fast else 0)
    attn_fn = dit.prefix_cached_attention(prefix_k, prefix_v, settings)
    return observed.original_block(self, x, mod, pe, attn_fn, 0, settings)


block_compilers, attention_compilers = {}, {}
for name in ('baseline', 'block', 'loop', 'block-loop', 'block-norm', 'block-loop-norm', 'native', 'block-native'):
    function = types.FunctionType(dit._attention.__code__.replace(co_name='d2_attention_' + name), dit.__dict__)
    attention_compilers[name] = torch.compile(function, backend='hpu_backend', dynamic=None,
                                             options={'force_static_compile': True})
    if name.startswith('block'):
        function = types.FunctionType(prefix_block.__code__.replace(co_name='d2_prefix_' + name), globals())
        block_compilers[name] = torch.compile(function, backend='hpu_backend', dynamic=None,
                                             options={'force_static_compile': True})
observed.baseline_compiler = lambda: attention_compilers[mode]


@functools.wraps(original_block)
def block(self, x, mod, pe, attn_fn, prefix_len, transformer_options={}):
    index = transformer_options['block_index']
    selected = (observed.capture and observed.forward_index in (7, 8) and index in (0, 15, 31)
                and x.shape[1] == 8160 and prefix_len == 0)
    saved = None
    if selected:
        closure = inspect.getclosurevars(attn_fn).nonlocals
        saved = dict(x=x.cpu(), pe=pe.cpu(),
                     mod=[tuple(t.cpu() for t in item) if isinstance(item, tuple) else item.cpu() for item in mod],
                     prefix_k=closure['prefix_k'].cpu(), prefix_v=closure['prefix_v'].cpu(),
                     layer=index, forward=observed.forward_index, prefix_len=prefix_len)
    if mode.startswith('block') and prefix_len == 0 and x.shape[1] >= 4096:
        closure = inspect.getclosurevars(attn_fn).nonlocals
        result = block_compilers[mode](self, x, mod, pe, closure['prefix_k'], closure['prefix_v'], index != 0)
    else:
        result = original_block(self, x, mod, pe, attn_fn, prefix_len, transformer_options)
    if saved is not None:
        saved['output'] = result.cpu()
        directory = root / 'blocks'
        directory.mkdir(exist_ok=True)
        torch.save(saved, directory / f'forward-{observed.forward_index:02d}-layer-{index:02d}.pt')
    return result


dit.QwenImage21TransformerBlock.forward = block
original_execute = observed.execution.PromptExecutor.execute_async


async def execute(self, prompt, prompt_id, *args, **kwargs):
    global mode, native_runtime
    extra = args[0] if args else kwargs.get('extra_data', {})
    mode = extra.get('d2_candidate', 'baseline')
    dit.attention = observed.attention if extra.get('d1_capture', False) else observed.original_attention
    attention_module._hpu_fast_attention = native_attention if 'native' in mode else loop_attention if 'loop' in mode else original_fast
    dit._modulated_norm = inlined_modulated_norm if mode.endswith('norm') else original_modulated_norm
    try:
        return await original_execute(self, prompt, prompt_id, *args, **kwargs)
    finally:
        if native_runtime is not None:
            (root / (prompt_id + '-native.json')).write_text(json.dumps(native_runtime.records, indent=2))
            native_runtime.close()
            native_runtime = None


observed.execution.PromptExecutor.execute_async = execute


def main():
    observed.main()


if __name__ == '__main__':
    main()
