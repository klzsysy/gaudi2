"""Isolated prefill/encoding regions; all outputs and preparation remain timed."""
import functools
import json
import os
from pathlib import Path
import runpy
import statistics
import time

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True
from comfy.text_encoders import qwen35
from comfy.text_encoders.llama import MLP

mode = os.environ.get('MOSS_PREFILL_SCREEN', 'product')
for name, cls in [('mlp', MLP), ('attention', qwen35.GatedAttention)]:
    if name not in mode and mode != 'both':
        continue
    original = cls.forward
    compiled = torch.compile(original, backend='hpu_backend', dynamic=None, options={'force_static_compile': True})
    def wrapper(original, compiled):
        @functools.wraps(original)
        def forward(module, x, *values, **kwargs):
            return (compiled if x.shape[1] > 8 else original)(module, x, *values, **kwargs)
        return forward
    cls.forward = wrapper(original, compiled)

state = runpy.run_path(str(Path(__file__).with_name('moss_gdn_experiment.py')), run_name='__main__')
clip = state['clip']
directory = state['directory']
caption = Path(os.environ['MOSS_CAPTION']).read_text()
results = []
for name, text in [('positive', caption), ('negative', state['workflow']['18:5']['inputs']['text'])]:
    tokens = clip.tokenize(text)
    times = []
    for index in range(6):
        torch.hpu.synchronize()
        start = time.perf_counter()
        output = clip.encode_from_tokens(tokens)
        torch.hpu.synchronize()
        times.append(time.perf_counter()-start)
    torch.save(output, directory / f'encoding-{name}.pt')
    results.append(dict(name=name, cold=times[0], median=statistics.median(times[1:]), warm=times[1:]))
(directory/'encoding-summary.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results), flush=True)
