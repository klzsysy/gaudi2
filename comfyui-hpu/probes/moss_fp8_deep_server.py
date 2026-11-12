"""Capture selected Moody block inputs separately from warm workflow timings."""
import collections
import os
from pathlib import Path
import runpy

import comfy.options
comfy.options.enable_args_parsing()
import torch
from comfy.ldm.moss_image import model as moss
import execution

if os.environ.get('MOSS_DEEP_SHARED_QUANT') == '1':
    import moss_fp8_shared_quant
    moss_fp8_shared_quant.install()

product_fp8 = False
if os.environ.get('MOSS_DEEP_ATTENTION_AB') == '1' or os.environ.get('MOSS_DEEP_PRODUCT_AB') == '1':
    import moss_fp8_attention_candidate
    baseline_heads = moss._attention_heads
    original_execute = execution.PromptExecutor.execute_async
    async def execute(self, prompt, prompt_id, *args, **kwargs):
        global product_fp8
        extra = args[0] if args else kwargs.get('extra_data', {})
        product_fp8 = extra.get('moss_fp8_attention', False)
        if product_fp8 and os.environ.get('MOSS_DEEP_ATTENTION_AB') == '1':
            moss_fp8_attention_candidate.install()
        else:
            moss._attention_heads = baseline_heads
        return await original_execute(self, prompt, prompt_id, *args, **kwargs)
    execution.PromptExecutor.execute_async = execute

if os.environ.get('MOSS_DEEP_ATTENTION') == '1':
    import moss_fp8_attention_candidate
    moss_fp8_attention_candidate.install()

directory = Path(os.environ['MOSS_PIPELINE_DIR'])
counts = collections.Counter()
context = None
original_forward = moss.MOSSImage._forward
original_block = moss.Block.forward

def forward(self, x, timestep, context_tensor, transformer_options={}, *args, **kwargs):
    global context
    if os.environ.get('MOSS_DEEP_PRODUCT_AB') == '1':
        transformer_options = dict(transformer_options, fp8_attention=product_fp8)
    length = context_tensor.shape[1]
    index = counts[length]
    counts[length] += 1
    capture = os.environ.get('MOSS_DEEP_CAPTURE') == '1' and index // 25 == 6 and index % 25 in (0, 12, 24)
    context = (length, index % 25, {id(block): layer for layer, block in enumerate(self.layers)}) if capture else None
    try:
        if os.environ.get('MOSS_DEEP_PROFILE') == '1' and index == 6 * 25 + 12:
            torch.hpu.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU]) as profile:
                output = original_forward(self, x, timestep, context_tensor, transformer_options, *args, **kwargs)
                torch.hpu.synchronize()
            profile.profiler.export_chrome_trace(str(directory / f'dit-{length}-step12.trace.json'))
            return output
        return original_forward(self, x, timestep, context_tensor, transformer_options, *args, **kwargs)
    finally:
        context = None

def block_forward(self, x, mod, pe, transformer_options={}, *args, **kwargs):
    if context is not None and context[2].get(id(self)) in (0, 20, 41):
        layer = context[2][id(self)]
        target = directory / 'captures'
        target.mkdir(exist_ok=True)
        torch.save(dict(x=x.cpu(), mod=tuple(value.cpu() for value in mod), pe=pe.cpu(),
                        blockwise=transformer_options.get('_moss_hpu_blockwise_rope', False), layer=layer),
                   target / f'input-{context[0]}-{context[1]}-{layer}.pt')
    return original_block(self, x, mod, pe, transformer_options, *args, **kwargs)

moss.MOSSImage._forward = forward
moss.Block.forward = block_forward
runpy.run_path(str(Path(__file__).with_name('moss_observed_server.py')), run_name='__main__')
