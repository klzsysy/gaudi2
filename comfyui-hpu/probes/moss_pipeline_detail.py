"""Synchronized diagnostic regions, enabled for a separate request only."""
import functools

import torch
import comfy.sd
from comfy.ldm.moss_image import model as moss
from comfy.text_encoders.llama import BaseGenerate
from comfy.text_encoders.qwen35 import MTPHead, Qwen35Transformer


def install(event, enabled):
    def measured(function, name):
        @functools.wraps(function)
        def invoke(*values, **kwargs):
            if not enabled():
                return function(*values, **kwargs)
            label = name(values, kwargs) if callable(name) else name
            torch.hpu.synchronize()
            event('region_start', region=label)
            try:
                return function(*values, **kwargs)
            finally:
                torch.hpu.synchronize()
                event('region_end', region=label)
        return invoke

    def text_phase(values, kwargs):
        embeds = kwargs.get('embeds')
        return 'text_prefill' if embeds is not None and embeds.shape[1] > 8 else 'text_main_decode'

    Qwen35Transformer.forward = measured(Qwen35Transformer.forward, text_phase)
    MTPHead.forward = measured(MTPHead.forward, 'text_mtp_head')
    BaseGenerate.sample_token = measured(BaseGenerate.sample_token, 'text_sampling')
    moss.MOSSImage._forward = measured(moss.MOSSImage._forward, 'dit_model_forward')
    moss.MOSSImage.rope = measured(moss.MOSSImage.rope, 'dit_rope')
    moss.TextEmbedder.forward = measured(moss.TextEmbedder.forward, 'dit_text_embedder')

    generate = comfy.sd.CLIP.generate
    @functools.wraps(generate)
    def wrapped_generate(clip, *values, **kwargs):
        if enabled():
            for name, module in clip.cond_stage_model.named_modules():
                if name.endswith('.lm_head') and not getattr(module, '_moss_timed_head', False):
                    module.forward = measured(module.forward, 'text_lm_head')
                    module._moss_timed_head = True
        return generate(clip, *values, **kwargs)
    comfy.sd.CLIP.generate = wrapped_generate

    original_block = moss.Block.forward
    @functools.wraps(original_block)
    def block(layer, x, mod, pe, transformer_options={}, *values, **kwargs):
        if not enabled():
            return original_block(layer, x, mod, pe, transformer_options, *values, **kwargs)
        if transformer_options.get('block_index') == 0:
            torch.hpu.synchronize()
            event('region_start', region='dit_blocks')
        result = original_block(layer, x, mod, pe, transformer_options, *values, **kwargs)
        if transformer_options.get('block_index') == transformer_options.get('total_blocks', -1) - 1:
            torch.hpu.synchronize()
            event('region_end', region='dit_blocks')
        return result
    moss.Block.forward = block
