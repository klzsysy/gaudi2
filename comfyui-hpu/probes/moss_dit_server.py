"""Investigation-only DiT patches; launch the existing observed server."""
import functools
import os
from pathlib import Path
import runpy

import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.ops
import comfy.ldm.moss_image.model as moss

directory = Path(os.environ['MOSS_DIT_DIRECTORY'])
mode = os.environ.get('MOSS_DIT_MODE', 'original')
if mode not in ('product', 'product-final', 'acceptance-final', 'block-split', 'situ-only') and hasattr(moss, '_attention'):
    moss.attention = moss._attention
if mode in ('product-contiguous', 'product-auto') and not hasattr(comfy, 'hpu_attention'):
    raise RuntimeError('This rejected candidate requires patches/rejected-shared-dynamic-sdpa.patch')
if mode == 'original' and hasattr(comfy, 'hpu_attention'):
    # Keep the previous GDN optimization while disabling the candidate SDPA region.
    comfy.hpu_attention.compiled_attention = lambda: torch.nn.functional.scaled_dot_product_attention
if mode == 'product-contiguous':
    sdpa_original = comfy.ops.scaled_dot_product_attention
    def contiguous_sdpa(q, k, v, *values, **kwargs):
        return sdpa_original(q.contiguous(), k.contiguous(), v.contiguous(), *values, **kwargs)
    comfy.ops.scaled_dot_product_attention = contiguous_sdpa
elif mode == 'product-auto':
    attention_auto = torch.compile(comfy.hpu_attention._attention, backend='hpu_backend', dynamic=None)
    comfy.hpu_attention.compiled_attention = lambda: attention_auto


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(x) for x in value)
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    return value


if mode.startswith('sdpa'):
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        return FusedSDPA.apply(q, k, v, attn_mask, dropout_p, is_causal, scale, 'None', mode == 'sdpa-recompute')
    comfy.ops.scaled_dot_product_attention = sdpa

if mode in ('block-fixed', 'block-decomposed'):
    def fp16_linear_wanted(x):
        return x.is_cuda and x.dtype == torch.float16 and getattr(torch.backends.cuda.matmul, 'allow_fp16_accumulation', False)
    comfy.ops._fp16_linear_wanted = fp16_linear_wanted
    if mode == 'block-decomposed':
        from habana_frameworks.torch.dynamo.compile_backend.decomposition import hpu_backend_decompositions_common
        from comfy_kitchen.backends.eager.adaln import rms_adaln
        from comfy_kitchen.backends.eager.rope import rms_rope_split_half
        hpu_backend_decompositions_common[torch.ops.comfy_kitchen.rms_adaln.default] = rms_adaln
        hpu_backend_decompositions_common[torch.ops.comfy_kitchen.rms_rope_split_half.default] = rms_rope_split_half

if mode in ('block-compile', 'block-fixed', 'block-decomposed'):
    moss.Block.forward = torch.compile(moss.Block.forward, backend='hpu_backend', fullgraph=True, dynamic=False)
elif mode in ('attention-region', 'attention-auto'):
    moss.attention = torch.compile(moss.attention, backend='hpu_backend', fullgraph=True, dynamic=None if mode == 'attention-auto' else False)
elif mode == 'ops-compile':
    for name in ('modulated_rms_norm', 'gated_residual', 'attention'):
        setattr(moss, name, torch.compile(getattr(moss, name), backend='hpu_backend', fullgraph=True, dynamic=False))
    comfy.ops.INPUT_ACT_EAGER['situ'] = torch.compile(comfy.ops.INPUT_ACT_EAGER['situ'], backend='hpu_backend', fullgraph=True, dynamic=False)
elif mode == 'block-split':
    import moss_block_candidate
    comfy.ops._fp16_linear_wanted = moss_block_candidate.fp16_wanted
    moss.Block.forward = moss_block_candidate.forward
elif mode == 'situ-only':
    comfy.ops.INPUT_ACT_EAGER['situ'] = torch.compile(comfy.ops._situ_eager, backend='hpu_backend', dynamic=None)

if os.environ.get('MOSS_DIT_CAPTURE') == '1':
    seen = set()
    forward = moss.MOSSImage._forward
    @functools.wraps(forward)
    def capture_model(self, x, timestep, context, transformer_options={}, **kwargs):
        key = (tuple(x.shape), tuple(context.shape))
        if key not in seen:
            index = len(seen)
            seen.add(key)
            torch.save(cpu({'x': x, 'timestep': timestep, 'context': context,
                            'transformer_options': transformer_options, 'kwargs': kwargs}), directory / f'model-input-{index}.pt')
            print('DIT_CAPTURE', key, transformer_options.keys(), flush=True)
        return forward(self, x, timestep, context, transformer_options, **kwargs)
    moss.MOSSImage._forward = capture_model
    block_forward = moss.Block.forward
    block_seen = set()
    @functools.wraps(block_forward)
    def capture_block(self, x, mod, pe, transformer_options={}, ref=None, ref_kv=None, kv_out=None, mask=None):
        key = tuple(x.shape)
        if transformer_options.get('block_index') == 0 and key not in block_seen:
            index = len(block_seen)
            block_seen.add(key)
            torch.save(cpu({'state': self.state_dict(), 'args': (x, mod, pe), 'mask': mask,
                            'heads': self.num_heads, 'fused_qkv': self.fused_qkv}), directory / f'block-input-{index}.pt')
        return block_forward(self, x, mod, pe, transformer_options, ref, ref_kv, kv_out, mask)
    moss.Block.forward = capture_block

if os.environ.get('MOSS_BLOCK_MULTI_CAPTURE'):
    import moss_block_capture

runpy.run_path(str(Path(__file__).with_name('moss_observed_server.py')), run_name='__main__')
