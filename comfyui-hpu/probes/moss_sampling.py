"""Isolate Qwen3.5 sampling's HPU int64 gather failure without model weights.

Run from the independent ComfyUI checkout while the server is stopped.
Default uses the fixed int32 gather. Set MOSS_ORIGINAL_GATHER=1 to reproduce
the original compile failure; use a separate process for that invocation.
"""
import os

import torch

from comfy.text_encoders.llama import BaseGenerate


logits = torch.linspace(-10, 10, 248320).reshape(1, -1).to("hpu")
probs, indices = BaseGenerate().processed_probs(logits, 0.7, 64, 0.95, 0.05, 1.0, [])
next_index = torch.multinomial(probs, 1, generator=torch.Generator(device="hpu").manual_seed(0))
print("top_indices", indices.cpu(), "sample_index", next_index.cpu(), flush=True)
expected = indices.cpu().gather(1, next_index.cpu())
if os.environ.get("MOSS_ORIGINAL_GATHER") == "1":
    actual = indices.gather(1, next_index)
else:
    actual = indices.int().gather(1, next_index).long()
torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
print("PASS", actual.cpu(), flush=True)
