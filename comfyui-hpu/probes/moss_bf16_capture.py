"""Capture real states in the first request and short warm device traces later."""
import collections
import functools
import os
from pathlib import Path

import torch
from comfy.ldm.moss_image import model as moss


directory = Path(os.environ['MOSS_BF16_CAPTURE'])
directory.mkdir(parents=True, exist_ok=True)
counts = collections.Counter()
saved_weights = set()
current = None
profiling = False
profile_request = int(os.environ.get('MOSS_BF16_PROFILE_REQUEST', '6'))
profile_steps = {int(v) for v in os.environ.get('MOSS_BF16_PROFILE_STEPS', '0').split(',')}


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(v) for v in value)
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    return value


original_model = moss.MOSSImage._forward
@functools.wraps(original_model)
def model_forward(self, x, timestep, context, transformer_options={}, **kwargs):
    global current, profiling
    length = context.shape[1]
    index = counts[length]
    counts[length] += 1
    current = (length, index)
    profiling = index // 25 == profile_request and index % 25 in profile_steps
    try:
        if not profiling:
            return original_model(self, x, timestep, context, transformer_options, **kwargs)
        torch.hpu.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
                                    record_shapes=False, with_stack=False) as profile:
            with torch.profiler.record_function(f'moss_forward_N{4096 + length}'):
                result = original_model(self, x, timestep, context, transformer_options, **kwargs)
            torch.hpu.synchronize()
        suffix = '' if index % 25 == 0 else f'-step{index % 25}'
        profile.profiler.export_chrome_trace(str(directory / f'forward-{length}{suffix}.trace.json'))
        (directory / f'forward-{length}{suffix}.cpu.txt').write_text(profile.key_averages().table(sort_by='self_cpu_time_total', row_limit=50))
        print('BF16_PROFILE_SAVED', length, flush=True)
        return result
    finally:
        current, profiling = None, False
moss.MOSSImage._forward = model_forward

original_block = moss.Block.forward
@functools.wraps(original_block)
def block_forward(self, x, mod, pe, transformer_options={}, ref=None, ref_kv=None, kv_out=None, mask=None):
    layer = transformer_options.get('block_index')
    if current is not None and current[1] == 0 and layer not in saved_weights and os.environ.get('MOSS_CAPTURE_ALL_WEIGHTS') == '1':
        torch.save(cpu(self.state_dict()), directory / f'weights-{layer}.pt')
        saved_weights.add(layer)
    if current is not None and current[1] in (0, 12, 24) and layer in (0, 20, 41):
        if layer not in saved_weights:
            torch.save(cpu(self.state_dict()), directory / f'weights-{layer}.pt')
            saved_weights.add(layer)
        torch.save(cpu({'args': (x, mod, pe), 'mask': mask, 'heads': self.num_heads,
                        'fused_qkv': self.fused_qkv, 'layer': layer, 'condition_length': current[0], 'step': current[1]}),
                   directory / f'input-{current[0]}-{current[1]}-{layer}.pt')
        print('BF16_CAPTURE', current, layer, tuple(x.shape), flush=True)
    if profiling:
        with torch.profiler.record_function(f'moss_block_{layer}'):
            return original_block(self, x, mod, pe, transformer_options, ref, ref_kv, kv_out, mask)
    return original_block(self, x, mod, pe, transformer_options, ref, ref_kv, kv_out, mask)
moss.Block.forward = block_forward
