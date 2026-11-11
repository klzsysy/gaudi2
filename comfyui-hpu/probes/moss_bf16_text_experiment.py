"""Capture or compile existing Qwen prefill regions in an isolated text process."""
import functools
import os
from pathlib import Path
import runpy

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True
import comfy.text_encoders.qwen35 as qwen
from comfy.text_encoders.llama import MLP

mode = os.environ.get('MOSS_PREFILL_MODE', 'product')
directory = Path(os.environ['MOSS_RESULT_DIR'])
directory.mkdir(parents=True, exist_ok=True)
original = qwen.torch_chunk_gated_delta_rule
seen = set()


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(v) for v in value)
    if isinstance(value, dict):
        return {k: cpu(v) for k,v in value.items()}
    return value


if mode == 'capture':
    @functools.wraps(original)
    def capture(*values, **kwargs):
        key = (values[0].shape[1], bool(kwargs.get('output_final_state')))
        inputs = cpu((values,kwargs)) if key not in seen else None
        result = original(*values,**kwargs)
        if inputs is not None:
            seen.add(key)
            torch.save({'args':inputs[0], 'kwargs':inputs[1], 'expected':cpu(result)}, directory/f'prefill-{key[0]}-{int(key[1])}.pt')
            print('PREFILL_CAPTURE',key,flush=True)
        return result
    qwen.torch_chunk_gated_delta_rule=capture
elif mode in ('delta','both'):
    qwen.torch_chunk_gated_delta_rule=torch.compile(original,backend='hpu_backend',dynamic=False,options={'force_static_compile':True})
if mode in ('mlp','both'):
    MLP.forward=torch.compile(MLP.forward,backend='hpu_backend',dynamic=None,options={'force_static_compile':True})

state=runpy.run_path(str(Path(__file__).with_name('moss_gdn_experiment.py')),run_name='__main__')
if mode=='capture':
    clip=state['clip']
    caption=Path(os.environ['MOSS_CAPTION']).read_text()
    workflow=state['workflow']
    for text in (caption, workflow['18:5']['inputs']['text']):
        tokens=clip.tokenize(text)
        clip.encode_from_tokens(tokens)
