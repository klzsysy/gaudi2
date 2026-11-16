"""Qwen FP8 stage traces and calibrated HBM capture in the isolated server."""
import functools
import json
import time

import torch
import qwen21_deep_server as deep
import execution
from moss_native_profile import NativeProfile

torch.profiler.profile = functools.partial(torch.profiler.profile, record_shapes=True)
forward = deep.dit.QwenImage21Transformer2DModel._forward


def denoise(self, *args, **kwargs):
    if deep.mode not in ("native_dit", "window_dit") or deep.dit_index != 3:
        return forward(self, *args, **kwargs)
    torch.hpu.synchronize()
    profiler = NativeProfile(deep.root) if deep.mode == "native_dit" else None
    if profiler:
        profiler.__enter__()
    start = time.perf_counter()
    result = forward(self, *args, **kwargs)
    torch.hpu.synchronize()
    seconds = time.perf_counter() - start
    if profiler:
        profiler.__exit__(None, None, None)
    filename = "window.json" if profiler else "plain-window.json"
    (deep.root / filename).write_text(json.dumps(dict(seconds=seconds, forwards=1)))
    return result


deep.dit.QwenImage21Transformer2DModel._forward = denoise


get_output = execution.get_output_data


async def output(prompt_id, unique_id, obj, *args, **kwargs):
    if deep.mode not in ("native_clip", "window_clip") or type(obj).__name__ != "TextEncodeQwenImage21":
        return await get_output(prompt_id, unique_id, obj, *args, **kwargs)
    torch.hpu.synchronize()
    profiler = NativeProfile(deep.root) if deep.mode == "native_clip" else None
    if profiler:
        profiler.__enter__()
    start = time.perf_counter()
    result = await get_output(prompt_id, unique_id, obj, *args, **kwargs)
    torch.hpu.synchronize()
    seconds = time.perf_counter() - start
    if profiler:
        profiler.__exit__(None, None, None)
    filename = "window.json" if profiler else "plain-window.json"
    (deep.root / filename).write_text(json.dumps(dict(seconds=seconds, forwards=2)))
    return result


execution.get_output_data = output


def main():
    deep.main()
