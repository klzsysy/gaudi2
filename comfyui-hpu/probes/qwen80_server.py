"""Ordinary node execution with request-scoped readback A/B selection."""
from pathlib import Path
import runpy

import comfy.options
comfy.options.enable_args_parsing()
import execution
from comfy.text_encoders import qwen35

selected_readback = 4
original_readback = qwen35.Qwen35._generation_readback_size


def readback(self, embeds, caches):
    return min(selected_readback, original_readback(self, embeds, caches))


qwen35.Qwen35._generation_readback_size = readback
original_execute = execution.PromptExecutor.execute_async


async def execute(self, prompt, prompt_id, *args, **kwargs):
    global selected_readback
    extra = args[0] if args else kwargs.get('extra_data', {})
    selected_readback = extra.get('qwen_readback', 4)
    assert selected_readback in (1, 4)
    return await original_execute(self, prompt, prompt_id, *args, **kwargs)


execution.PromptExecutor.execute_async = execute


def main():
    runpy.run_path(str(Path(__file__).with_name('moss_observed_server.py')), run_name='__main__')


if __name__ == '__main__':
    main()
