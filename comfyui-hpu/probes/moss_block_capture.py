"""Capture early/middle/late DiT states for three depths in one real request."""
import functools
import os
from pathlib import Path

import torch
from comfy.ldm.moss_image import model as moss

directory = Path(os.environ['MOSS_BLOCK_MULTI_CAPTURE'])
directory.mkdir(parents=True, exist_ok=True)
positive_count = 0
current = None
saved_weights = set()


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(x) for x in value)
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    return value


forward = moss.MOSSImage._forward
@functools.wraps(forward)
def capture_forward(self, x, timestep, context, transformer_options={}, **kwargs):
    global positive_count, current
    if context.shape[1] > 100:
        current = positive_count
        positive_count += 1
    else:
        current = None
    try:
        return forward(self, x, timestep, context, transformer_options, **kwargs)
    finally:
        current = None
moss.MOSSImage._forward = capture_forward

block_forward = moss.Block.forward
@functools.wraps(block_forward)
def capture_block(self, x, mod, pe, transformer_options={}, ref=None, ref_kv=None, kv_out=None, mask=None):
    layer = transformer_options.get('block_index')
    if current in (0, 12, 24) and layer in (0, 20, 41):
        if layer not in saved_weights:
            torch.save(cpu(self.state_dict()), directory / f'weights-{layer}.pt')
            saved_weights.add(layer)
        torch.save(cpu({'args': (x, mod, pe), 'heads': self.num_heads, 'mask': mask, 'layer': layer, 'step': current}), directory / f'input-{current}-{layer}.pt')
        print('BLOCK_CAPTURE', current, layer, tuple(x.shape), flush=True)
    return block_forward(self, x, mod, pe, transformer_options, ref, ref_kv, kv_out, mask)
moss.Block.forward = capture_block
