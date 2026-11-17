"""Interleaved ordinary-node validation of request-local PE optimizations."""
import ast
import functools
import json
import os
from pathlib import Path
import runpy

import comfy.options
comfy.options.enable_args_parsing()
import execution
import torch
from comfy.text_encoders import qwen35, llama, hpu_kv_attention

baseline = False
teacher = None
sample_index = 0
prompt_id = None
root = Path(os.environ['QWEN80_VALIDATION'])
tree = ast.parse((root / 'baseline-source/comfy/text_encoders/qwen35.py').read_text())
owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Qwen35')
factory = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == '_generation_forward')
namespace = dict(vars(qwen35))
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
    if teacher is not None:
        result = torch.full_like(result, teacher[min(sample_index, len(teacher) - 1)])
    sample_index += 1
    return result


llama.BaseGenerate.sample_token = sampled
original_generate = llama.BaseGenerate.generate


@functools.wraps(original_generate)
def generate(self, *values, **kwargs):
    result = original_generate(self, *values, **kwargs)
    (root / 'api' / (prompt_id + '-tokens.json')).write_text(json.dumps(result))
    return result


llama.BaseGenerate.generate = generate
original_execute = execution.PromptExecutor.execute_async


async def execute(self, prompt, current_prompt_id, *values, **kwargs):
    global baseline, teacher, sample_index, prompt_id
    extra = values[0] if values else kwargs.get('extra_data', {})
    baseline = extra.get('qwen_baseline', False)
    teacher = extra.get('qwen_teacher')
    sample_index = 0
    prompt_id = current_prompt_id
    return await original_execute(self, prompt, current_prompt_id, *values, **kwargs)


execution.PromptExecutor.execute_async = execute


def main():
    runpy.run_path(str(Path(__file__).with_name('moss_observed_server.py')), run_name='__main__')


if __name__ == '__main__':
    main()
