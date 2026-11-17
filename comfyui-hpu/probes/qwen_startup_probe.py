"""Measure Qwen decode compilation across prompt and capacity changes."""
import argparse
import inspect
import json
from pathlib import Path
import textwrap
import time

import torch

from comfy.cli_args import args

args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True

import comfy.sd
from comfy.text_encoders import qwen35
from comfy.text_encoders.llama import BaseGenerate
from torch._dynamo.backends.registry import lookup_backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--mode', choices=['baseline', 'dynamic', 'dynamic_compact'], required=True)
    parser.add_argument('--history', type=Path, required=True)
    arguments = parser.parse_args()
    arguments.root.mkdir(parents=True)
    source = textwrap.dedent(inspect.getsource(qwen35.Qwen35._generation_forward))
    if 'mark_dynamic' in source:
        raise RuntimeError('Screening requires the recorded staged.87ea451fce53 baseline source')
    if arguments.mode == 'dynamic_compact':
        old = 'caches[0].index + 1].clone()'
        assert old in source
        source = source.replace(old, 'caches[0].index + 1].clone(memory_format=torch.contiguous_format)')
        namespace = dict(vars(qwen35))
        exec(compile(source, '<compact-rotary-input>', 'exec'), namespace)
        qwen35.Qwen35._generation_forward = namespace['_generation_forward']
    torch.set_grad_enabled(False)
    torch.set_num_threads(32)
    history = json.loads(arguments.history.read_text())
    graphs = [item['prompt'][2] for item in history.values()]
    source = graphs[0]
    clip = comfy.sd.load_clip(
        ['/home/sonny/ComfyUI-Models/models/text_encoders/' + source['459:473']['inputs']['clip_name']],
        clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
    backend = lookup_backend('hpu_backend')
    compilations = []

    def counted_backend(graph, inputs, **kwargs):
        start = time.perf_counter()
        compiled = backend(graph, inputs, **kwargs)
        compilations.append(time.perf_counter() - start)
        print('DECODE_COMPILED', compilations[-1], flush=True)
        return compiled

    compiled = torch.compile(qwen35._hpu_decode, backend=counted_backend, fullgraph=True,
                             dynamic=None if arguments.mode.startswith('dynamic') else False,
                             options={'force_static_compile': True})

    def decode(model, hidden, rotary, caches):
        if arguments.mode.startswith('dynamic'):
            for cache in caches:
                if isinstance(cache, qwen35.HPUSingleTokenKV):
                    torch._dynamo.mark_dynamic(cache.key, 2)
                    torch._dynamo.mark_dynamic(cache.value, 2)
        return compiled(model, hidden, rotary, caches)

    qwen35._compiled_hpu_decode = lambda: decode
    if arguments.mode == 'dynamic_compact':
        namespace['_compiled_hpu_decode'] = qwen35._compiled_hpu_decode
    sample = BaseGenerate.sample_token
    counter = 0
    checkpoints = []
    started = 0

    def sampled(self, *values, **kwargs):
        nonlocal counter
        result = sample(self, *values, **kwargs)
        counter += 1
        if counter <= 4:
            torch.hpu.synchronize()
            checkpoints.append(time.perf_counter() - started)
        if counter == 64:
            result = torch.full_like(result, self.model.config.stop_tokens[0])
        return result

    BaseGenerate.sample_token = sampled
    cases = [('original', graphs[0], None)]
    cases += [(str(index), graph, None) for index, graph in enumerate(graphs[-6:])]
    cases += [('capacity-' + str(capacity), graphs[0], capacity) for capacity in [4095, 4097, 8192, 8193, 16384]]
    cases += [('repeat', graphs[0], None)]
    rows = []
    for label, graph, max_length in cases:
        settings = graph['459:471']['inputs']
        tokens = clip.tokenize(settings['prompt'], system_prompt=graph['459:475']['inputs']['value'],
                               min_length=1, thinking=settings['thinking'])
        clip.load_model(tokens)
        counter = 0
        checkpoints = []
        torch.hpu.synchronize()
        started = time.perf_counter()
        ids = clip.generate(tokens, do_sample=True, max_length=max_length or settings['max_length'],
                            temperature=1.0, top_k=20, top_p=.95, min_p=0., repetition_penalty=1.,
                            presence_penalty=1.5, seed=settings['sampling_mode.seed'], mtp=True)
        torch.hpu.synchronize()
        row = dict(label=label, seconds=time.perf_counter() - started, first_samples=checkpoints,
                   compilations=list(compilations), tokens=len(ids))
        rows.append(row)
        (arguments.root / 'timings.json').write_text(json.dumps(rows, indent=2))
        (arguments.root / (label + '-tokens.json')).write_text(json.dumps(ids))
        print('RESULT', row, flush=True)


if __name__ == '__main__':
    main()
