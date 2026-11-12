"""Install process-local candidates before the existing full-workflow observer."""
import os
from pathlib import Path
import runpy

import comfy.options
comfy.options.enable_args_parsing()

if os.environ.get('MOSS_TEXT_ROPE_BASELINE') == '1':
    from comfy.text_encoders import hpu_kv_attention
    hpu_kv_attention._compiled_rope = lambda: hpu_kv_attention.apply_rope_split_half

if os.environ.get('MOSS_ATTENTION_DATAFLOW'):
    from moss_attention_dataflow_candidate import install
    install(os.environ['MOSS_ATTENTION_DATAFLOW'])

if os.environ.get('MOSS_ATTENTION_DATAFLOW_AB') == '1':
    import execution
    from moss_attention_dataflow_candidate import install
    original_execute = execution.PromptExecutor.execute_async

    async def execute_request(self, prompt, prompt_id, *args, **kwargs):
        extra = args[0] if args else kwargs.get('extra_data', {})
        install(extra.get('attention_dataflow', 'baseline'))
        return await original_execute(self, prompt, prompt_id, *args, **kwargs)

    execution.PromptExecutor.execute_async = execute_request

if os.environ.get('MOSS_TEXT_ROPE'):
    import moss_text_rope_candidate
    moss_text_rope_candidate.install(os.environ['MOSS_TEXT_ROPE'])

server = 'moss_fp8_deep_server.py' if os.environ.get('MOSS_PIPELINE_DIR') else 'moss_observed_server.py'
runpy.run_path(str(Path(__file__).with_name(server)), run_name='__main__')
