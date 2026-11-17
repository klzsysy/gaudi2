"""Ordinary-node A/B validation of decoder graph reuse and first-step latency."""
import ast
import functools
import json
import os
from pathlib import Path
import runpy
import time

import comfy.options
comfy.options.enable_args_parsing()
import execution
import torch
from comfy.text_encoders import qwen35, llama
from torch._dynamo.backends.registry import lookup_backend

root = Path(os.environ['QWEN_STARTUP_VALIDATION'])
backend = lookup_backend('hpu_backend')
compilations = {'baseline': [], 'product': []}
baseline = False
prompt_id = None
sample_index = 0
sample_times = []
started = 0
measure_startup = True


def baseline_backend(graph, inputs, **kwargs):
    start = time.perf_counter()
    result = backend(graph, inputs, **kwargs)
    compilations['baseline'].append(time.perf_counter() - start)
    return result


def product_backend(graph, inputs, **kwargs):
    start = time.perf_counter()
    result = backend(graph, inputs, **kwargs)
    compilations['product'].append(time.perf_counter() - start)
    return result


baseline_compiled = torch.compile(qwen35._hpu_decode, backend=baseline_backend, fullgraph=True,
                                  dynamic=False, options={'force_static_compile': True})
compile_function = torch.compile


def counted_compile(function, **kwargs):
    if function is qwen35._hpu_decode:
        kwargs['backend'] = product_backend
    return compile_function(function, **kwargs)


torch.compile = counted_compile
tree = ast.parse((root / 'baseline-source/comfy/text_encoders/qwen35.py').read_text())
owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Qwen35')
factory = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == '_generation_forward')
namespace = dict(vars(qwen35), _compiled_hpu_decode=lambda: baseline_compiled)
exec(compile(ast.Module(body=[factory], type_ignores=[]), '<baseline-generation-factory>', 'exec'), namespace)
baseline_factory = namespace['_generation_forward']
product_factory = qwen35.Qwen35._generation_forward


def select_factory(self, embeds, caches):
    return (baseline_factory if baseline else product_factory)(self, embeds, caches)


qwen35.Qwen35._generation_forward = select_factory
original_sample = llama.BaseGenerate.sample_token


@functools.wraps(original_sample)
def sampled(self, *values, **kwargs):
    global sample_index
    result = original_sample(self, *values, **kwargs)
    sample_index += 1
    if sample_index <= 4:
        if measure_startup:
            torch.hpu.synchronize()
        sample_times.append(time.perf_counter() - started)
    return result


llama.BaseGenerate.sample_token = sampled
original_generate = llama.BaseGenerate.generate


@functools.wraps(original_generate)
def generate(self, *values, **kwargs):
    global started
    started = time.perf_counter()
    result = original_generate(self, *values, **kwargs)
    record = dict(baseline=baseline, tokens=result, sample_times=sample_times,
                  seconds=time.perf_counter() - started, compilations=compilations,
                  synchronized_first_samples=measure_startup)
    (root / 'api' / (prompt_id + '-generation.json')).write_text(json.dumps(record))
    return result


llama.BaseGenerate.generate = generate
original_execute = execution.PromptExecutor.execute_async


async def execute(self, prompt, current_prompt_id, *values, **kwargs):
    global baseline, sample_index, sample_times, prompt_id, measure_startup
    extra = values[0] if values else kwargs.get('extra_data', {})
    baseline = extra.get('qwen_baseline', False)
    label = prompt['461']['inputs']['filename_prefix'].rsplit('/', 1)[-1]
    measure_startup = extra.get('qwen_measure_startup', not label.startswith('warm'))
    sample_index = 0
    sample_times = []
    prompt_id = current_prompt_id
    return await original_execute(self, prompt, current_prompt_id, *values, **kwargs)


execution.PromptExecutor.execute_async = execute


def main():
    runpy.run_path(str(Path(__file__).with_name('moss_observed_server.py')), run_name='__main__')


if __name__ == '__main__':
    main()
