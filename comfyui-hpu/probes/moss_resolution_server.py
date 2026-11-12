"""Select the previous attention cutoff and capture long-sequence block inputs."""
from collections import Counter
import os
from pathlib import Path
import runpy

import comfy.options
comfy.options.enable_args_parsing()
import execution
import torch
from comfy.ldm.moss_image import model as moss


root = Path(os.environ["MOSS_PIPELINE_DIR"])
settings = {}
counts = Counter()
context = None
execute = execution.PromptExecutor.execute_async
forward = moss.MOSSImage._forward
block_forward = moss.Block.forward


async def execute_request(self, prompt, prompt_id, *args, **kwargs):
    global settings
    settings = args[0] if args else kwargs.get("extra_data", {})
    counts.clear()
    return await execute(self, prompt, prompt_id, *args, **kwargs)


def model_forward(self, x, timestep, context_tensor, transformer_options={}, *args, **kwargs):
    global context
    length = context_tensor.shape[1]
    step = counts[length]
    counts[length] += 1
    if settings.get("resolution_legacy") and x.shape[-2] * x.shape[-1] + length > 8192:
        transformer_options = dict(transformer_options, fp8_attention=False)
    if settings.get("resolution_capture") and step in (0, 12, 24):
        context = (length, step, {id(block): layer for layer, block in enumerate(self.layers)})
    else:
        context = None
    try:
        return forward(self, x, timestep, context_tensor, transformer_options, *args, **kwargs)
    finally:
        context = None


def capture_block(self, x, mod, pe, transformer_options={}, *args, **kwargs):
    if context is not None and context[2].get(id(self)) in (0, 20, 41):
        layer = context[2][id(self)]
        target = root / "captures"
        target.mkdir(exist_ok=True)
        torch.save(dict(x=x.cpu(), mod=tuple(value.cpu() for value in mod), pe=pe.cpu(), layer=layer,
                        blockwise=transformer_options.get("_moss_hpu_blockwise_rope", False)),
                   target / f"input-{context[0]}-{context[1]}-{layer}.pt")
    return block_forward(self, x, mod, pe, transformer_options, *args, **kwargs)


execution.PromptExecutor.execute_async = execute_request
moss.MOSSImage._forward = model_forward
moss.Block.forward = capture_block
runpy.run_path(str(Path(__file__).with_name("moss_observed_server.py")), run_name="__main__")
