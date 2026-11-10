"""Run the existing text experiment with KV candidates, capture or profiler ranges."""
import collections
import json
import os
from pathlib import Path
import runpy
import time

import torch
from comfy.cli_args import args

args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True

from comfy.text_encoders.llama import FixedKVCache
from comfy.sd import CLIP
from moss_kv_candidate import attend as candidate_attend, install
import moss_decode_regions


directory = Path(os.environ['MOSS_RESULT_DIR'])
directory.mkdir(parents=True, exist_ok=True)
mode = os.environ.get('MOSS_KV_MODE', 'baseline')
if os.environ.get('MOSS_DISABLE_ATTENTION_REGION') == '1':
    from comfy.text_encoders.qwen35 import GatedAttention
    GatedAttention.forward = GatedAttention._forward
moss_decode_regions.install()
install(mode)
original = FixedKVCache.attend
counts = collections.Counter()
capture = os.environ.get('MOSS_KV_CAPTURE') == '1'
measure = os.environ.get('MOSS_KV_MEASURE') == '1'
stats = {}


def observed(cache, q, k, v):
    topology = cache.tracker['tree']
    label = 'plain' if topology is None else 'tree'
    signature = (label, tuple(q.shape), tuple(cache.key.shape))
    if counts[signature] == 0:
        print('KV_SHAPE', json.dumps({'mode': mode, 'q': list(q.shape), 'kv': list(cache.key.shape), 'index': cache.index, 'tree': topology is not None}), flush=True)
        if capture:
            tensors = {'q': q, 'key': cache.key, 'value': cache.value, 'length': cache.seqlen, 'k': k, 'v': v,
                       'mask': None if topology is None else topology.mask}
            torch.save({**{name: t.detach().cpu().clone() if t is not None else None for name, t in tensors.items()},
                        'index': cache.index}, directory / f'kv-{label}-{q.shape[2]}.pt')
    counts[signature] += 1
    if measure:
        torch.hpu.synchronize()
        start = time.perf_counter()
    with torch.profiler.record_function(f'moss_kv_{label}'):
        result = original(cache, q, k, v)
    if measure:
        torch.hpu.synchronize()
        key = str(signature)
        entry = stats.setdefault(key, {'calls': 0, 'seconds': 0.0})
        entry['calls'] += 1
        entry['seconds'] += time.perf_counter() - start
    return result


if os.environ.get('MOSS_KV_OBSERVE', '1') == '1' and not os.environ.get('MOSS_DECODE_REGIONS'):
    FixedKVCache.attend = observed
minimum_capacity = int(os.environ.get('MOSS_KV_CAPACITY', '0'))
if minimum_capacity:
    zeros = FixedKVCache.zeros.__func__
    def with_capacity(cls, batch, kv_heads, capacity, head_dim, device, dtype, shared):
        return zeros(cls, batch, kv_heads, max(capacity, minimum_capacity), head_dim, device, dtype, shared)
    FixedKVCache.zeros = classmethod(with_capacity)
if os.environ.get('MOSS_KV_ROTATE'):
    generate = CLIP.generate
    rotation = os.environ['MOSS_KV_ROTATE'].split(',')
    rotation_history = []
    baseline_attend = original
    def rotated_generate(clip, *values, **kwargs):
        global mode, original
        mode = rotation[len(rotation_history) % len(rotation)]
        original = baseline_attend if mode == 'baseline' else lambda cache, q, k, v: candidate_attend(cache, q, k, v, mode)
        rotation_history.append(mode)
        (directory / 'rotation-modes.json').write_text(json.dumps(rotation_history))
        return generate(clip, *values, **kwargs)
    CLIP.generate = rotated_generate
if os.environ.get('MOSS_KV_PROFILE') == '1':
    generate = CLIP.generate
    generations = 0
    def profiled_generate(clip, *values, **kwargs):
        global generations
        generations += 1
        if generations != 2:
            return generate(clip, *values, **kwargs)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
                                    record_shapes=False, with_stack=False) as profile:
            result = generate(clip, *values, **kwargs)
            torch.hpu.synchronize()
        profile.profiler.export_chrome_trace(str(directory / 'trace.json'))
        (directory / 'cpu-operators.txt').write_text(profile.key_averages().table(sort_by='self_cpu_time_total', row_limit=40))
        return result
    CLIP.generate = profiled_generate
try:
    runpy.run_path(str(Path(__file__).with_name('moss_gdn_experiment.py')), run_name='__main__')
finally:
    (directory / 'kv-stats.json').write_text(json.dumps({'mode': mode, 'counts': {str(k): v for k, v in counts.items()}, 'synchronized_measurements': stats}, indent=2))
