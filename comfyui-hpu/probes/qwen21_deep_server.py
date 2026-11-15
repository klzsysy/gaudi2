"""Qwen BF16 workflow diagnostics, isolated from normal product serving."""
import collections
import functools
import faulthandler
import gc
import json
import inspect
import os
from pathlib import Path
import runpy
import signal
import time

import comfy.options
comfy.options.enable_args_parsing()
import execution
import torch
from comfy.ldm.qwen_image21 import model as dit
from comfy.text_encoders import qwen35, hpu_mlp
from comfy.text_encoders.llama import BaseGenerate

root = Path(os.environ['QWEN_DEEP_DIR'])
mode = ''
regions = collections.defaultdict(lambda: [0, 0.0])
text_index = 0
dit_index = 0
active_profile = None
finished_profiles = []
window_start = 0
window_seconds = None
teacher_tokens = None

if os.environ.get('QWEN21_COMPARE_PRODUCT') == '1':
    from qwen21_deep_quality import install
    install()

if os.environ.get('QWEN21_TRIAL'):
    from qwen21_deep_candidates import install
    install(os.environ['QWEN21_TRIAL'])


def timed(function, label):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        name = label(args, kwargs) if callable(label) else label
        if mode == 'stages':
            torch.hpu.synchronize()
            start = time.perf_counter()
            result = function(*args, **kwargs)
            torch.hpu.synchronize()
            regions[name][0] += 1
            regions[name][1] += time.perf_counter() - start
            return result
        if active_profile:
            with torch.profiler.record_function(name):
                return function(*args, **kwargs)
        return function(*args, **kwargs)
    return wrapped


def text_label(values, kwargs):
    embeds = kwargs.get('embeds')
    return 'text_prefill' if embeds is not None and embeds.shape[1] > 8 else 'text_decode'


qwen35.Qwen35Transformer.forward = timed(qwen35.Qwen35Transformer.forward, text_label)
qwen35.GatedDeltaNet.forward = timed(qwen35.GatedDeltaNet.forward, lambda values, kwargs: 'decode_GDN' if values[1].shape[1] == 1 else 'prefill_GDN')
qwen35.GatedAttention.forward = timed(qwen35.GatedAttention.forward, lambda values, kwargs: 'decode_attention' if values[1].shape[1] == 1 else 'prefill_attention')
hpu_mlp.forward = timed(hpu_mlp.forward, 'decode_MLP')
BaseGenerate.logits = timed(BaseGenerate.logits, 'text_lm_head')
BaseGenerate.sample_token = timed(BaseGenerate.sample_token, 'text_sampling')
dit.QwenImage21Transformer2DModel.build_sequence = timed(dit.QwenImage21Transformer2DModel.build_sequence, 'dit_sequence')
dit.Attention.forward = timed(dit.Attention.forward, 'dit_attention_with_projections')
dit.SwiGLUFeedForward.forward = timed(dit.SwiGLUFeedForward.forward, 'dit_ffn')
dit._modulated_norm = timed(dit._modulated_norm, 'dit_modulation')
dit._gated_residual = timed(dit._gated_residual, 'dit_residual')
# The attention dispatcher is timed at its model-local call site.
if hasattr(dit, 'attention'):
    dit.attention = timed(dit.attention, 'dit_sdpa')
else:
    dit.optimized_attention = timed(dit.optimized_attention, 'dit_sdpa')

original_sample = BaseGenerate.sample_token

def sampled(self, *args, **kwargs):
    global text_index, active_profile, window_start, window_seconds
    result = original_sample(self, *args, **kwargs)
    if teacher_tokens is not None:
        result = torch.full_like(result, teacher_tokens[text_index])
    text_index += 1
    if mode in ('trace_text', 'window_text'):
        if text_index == 32:
            torch.hpu.synchronize()
            if mode == 'trace_text':
                active_profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU])
                active_profile.__enter__()
            window_start = time.perf_counter()
        if text_index == 48:
            torch.hpu.synchronize()
            window_seconds = time.perf_counter() - window_start
            if active_profile:
                active_profile.__exit__(None, None, None)
                finished_profiles.append(('text', active_profile))
                active_profile = None
    return result
BaseGenerate.sample_token = sampled

original_dit = dit.QwenImage21Transformer2DModel._forward

def denoise(self, *args, **kwargs):
    global dit_index, active_profile, window_seconds
    dit_index += 1
    if mode == 'trace_dit' and dit_index == 4:
        torch.hpu.synchronize()
        active_profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU])
        active_profile.__enter__()
        start = time.perf_counter()
        result = original_dit(self, *args, **kwargs)
        torch.hpu.synchronize()
        window_seconds = time.perf_counter()-start
        active_profile.__exit__(None, None, None)
        finished_profiles.append(('dit', active_profile))
        active_profile = None
        return result
    return original_dit(self, *args, **kwargs)
dit.QwenImage21Transformer2DModel._forward = timed(denoise, 'dit_forward')


original_block = dit.QwenImage21TransformerBlock.forward

def capture_block(self, x, mod, pe, attn_fn, prefix_len, transformer_options={}):
    if mode == 'capture' and dit_index == 4 and transformer_options.get('block_index') == 16:
        closure = inspect.getclosurevars(attn_fn).nonlocals
        capture = dict(state={k:v.detach().cpu() for k,v in self.state_dict().items()},x=x.detach().cpu(),
                       mod=[tuple(v.detach().cpu() for v in item) if isinstance(item,tuple) else item.detach().cpu() for item in mod],
                       pe=pe.detach().cpu(),prefix_len=prefix_len,heads=self.attn.heads,
                       prefix_k=closure['prefix_k'].detach().cpu(),prefix_v=closure['prefix_v'].detach().cpu())
        torch.save(capture, root/'block.pt')
    return original_block(self,x,mod,pe,attn_fn,prefix_len,transformer_options)
dit.QwenImage21TransformerBlock.forward = capture_block

original_execute = execution.PromptExecutor.execute_async

async def execute(self, prompt, prompt_id, *args, **kwargs):
    global mode, text_index, dit_index, window_seconds, teacher_tokens
    extra = args[0] if args else kwargs.get('extra_data', {})
    mode = extra.get('qwen_profile', '')
    teacher_tokens = extra.get('qwen_teacher')
    text_index = dit_index = 0
    window_seconds = None
    regions.clear()
    gc_enabled = gc.isenabled()
    if mode.startswith('trace') or mode == 'window_text':
        gc.collect()
        gc.disable()
    try:
        return await original_execute(self, prompt, prompt_id, *args, **kwargs)
    finally:
        teacher_tokens = None
        if gc_enabled:
            gc.enable()
        if mode:
            with (root/'diagnostics.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(prompt_id=prompt_id, mode=mode, regions=dict(regions),
                                             text_samples=text_index, dit_steps=dit_index, window_seconds=window_seconds))+'\n')
        for label, profile in finished_profiles:
            profile.profiler.export_chrome_trace(str(root/f'{prompt_id}-{label}.trace.json'))
            (root/f'{prompt_id}-{label}-operators.txt').write_text(profile.key_averages().table(sort_by='self_cpu_time_total',row_limit=45))
        finished_profiles.clear()
execution.PromptExecutor.execute_async = execute
def main():
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    runpy.run_path(str(Path(__file__).with_name('moss_observed_server.py')), run_name='__main__')


if __name__ == '__main__':
    main()
