"""Matched whole-workflow profiles and opt-in native FP8 text experiments."""
import collections
import functools
import os
from pathlib import Path
import runpy

import comfy.options
comfy.options.enable_args_parsing()
import torch
from comfy.ldm.moss_image import model as moss

if os.environ.get('MOSS_FP8_TEXT') == '1':
    import moss_fp8_text_candidate
    moss_fp8_text_candidate.install()

root = Path(os.environ['MOSS_PIPELINE_DIR'])
counts = collections.Counter()
original_forward = moss.MOSSImage._forward
@functools.wraps(original_forward)
def forward(self, x, timestep, context, *args, **kwargs):
    length = context.shape[1]
    index = counts[length]
    counts[length] += 1
    if os.environ.get("MOSS_SYSTEM_PROFILE", "1") != "1" or index // 25 != 6 or index % 25 not in (0, 12, 24):
        return original_forward(self, x, timestep, context, *args, **kwargs)
    torch.hpu.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU]) as profile:
        output = original_forward(self, x, timestep, context, *args, **kwargs)
        torch.hpu.synchronize()
    profile.profiler.export_chrome_trace(str(root / f'dit-{length}-step{index % 25}.trace.json'))
    (root / f'dit-{length}-step{index % 25}.cpu.txt').write_text(profile.key_averages().table(sort_by='self_cpu_time_total', row_limit=45))
    return output
moss.MOSSImage._forward = forward
runpy.run_path(str(Path(__file__).with_name('moss_observed_server.py')), run_name='__main__')
