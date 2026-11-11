"""Check shortlisted BF16 candidates across actual depths, steps, and CFG branches."""
import json
import os
from pathlib import Path

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True
import comfy.ops
from comfy.ldm.moss_image import model as moss
import moss_bf16_candidates as candidates
import habana_frameworks.torch.utils.debug as debug

torch.set_grad_enabled(False)
torch.set_num_threads(16)
root = Path(os.environ['MOSS_BF16_ROOT'])
records = []


def error(actual, expected):
    actual = actual.double()
    delta = actual - expected
    return {'rrmse': float(delta.square().mean().sqrt()/expected.square().mean().sqrt()),
            'max_abs': float(delta.abs().max()), 'finite': bool(actual.isfinite().all())}


for mode in os.environ.get('MOSS_BF16_CASES', 'product,adaln,rotary,native-all').split(','):
    torch._dynamo.reset()
    moss._compiled_block_ops.cache_clear()
    moss._compiled_attention.cache_clear()
    candidates.install(mode)
    for layer in (0, 20, 41):
        state = torch.load(root / f'captures/weights-{layer}.pt', weights_only=True)
        block = moss.Block(3072, 8192, 24, fused_qkv=True, dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
        block.load_state_dict(state)
        block.to('hpu')
        for condition in (307, 23):
            for step in (0, 12, 24):
                signature = f'{condition}-{step}-{layer}'
                data = torch.load(root / f'captures/input-{signature}.pt', weights_only=True)
                x = data['args'][0].to('hpu')
                mod = tuple(t.to('hpu') for t in data['args'][1])
                pe = data['args'][2].to('hpu')
                options = {}
                if os.environ.get('MOSS_VALIDATE_BLOCKWISE') == '1':
                    assert moss.hpu.blockwise_coefficients(pe)
                    options['_moss_hpu_blockwise_rope'] = True
                actual = block(x, mod, pe, options).cpu()
                expected = torch.load(root / f'fp64-{signature}.pt', weights_only=True)
                record = {'mode': mode, 'condition': condition, 'step': step, 'layer': layer,
                          'fp64_error': error(actual, expected), 'blockwise': bool(options),
                          'fallback': debug._get_fallback_op_count()}
                records.append(record)
                (root / os.environ.get('MOSS_VALIDATION_RESULTS', 'native-validation.json')).write_text(json.dumps(records, indent=2))
                print(json.dumps(record), flush=True)
        del block
