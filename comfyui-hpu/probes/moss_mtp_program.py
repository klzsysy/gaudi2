"""Record/replay MTP decisions while retaining sampling and draft computation."""
import json
import os
from pathlib import Path
import runpy

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True
import comfy.sd
from comfy.text_encoders import qwen35, hpu_gated_delta
from comfy.text_encoders.llama import BaseGenerate
if os.environ.get('MOSS_TREE_BASELINE') == '1':
    import moss_gdn_baseline
    moss_gdn_baseline.install()
if os.environ.get('MOSS_REGION_BASELINE') == '1':
    import moss_gdn_baseline
    moss_gdn_baseline.disable_region()
if os.environ.get('MOSS_DELTA_STATIC') == '1':
    import moss_delta_static
    moss_delta_static.install()
if os.environ.get('MOSS_PREFILL_EAGER') == '1':
    hpu_gated_delta.compiled_prefill = lambda function: function

root = Path(os.environ['MOSS_RESULT_DIR'])
root.mkdir(parents=True, exist_ok=True)
recording = os.environ.get('MOSS_MTP_RECORD') == '1'
program_path = Path(os.environ.get('MOSS_MTP_PROGRAM', str(root / 'program.pt')))
program = None if recording else torch.load(program_path, weights_only=True)
if program is not None:
    program['top'] = [x.to('hpu') for x in program['top']]
    program['sample'] = [x.to('hpu') for x in program['sample']]
active = False
indices = {}
captured = {}
control_seen = set()
generation = 0
profiler = None
original_topk = torch.Tensor.topk
original_sample = BaseGenerate.sample_token
original_widths = qwen35.best_widths
original_prepare = qwen35.LinearKV.prepare
original_generate = comfy.sd.CLIP.generate


def topk(tensor, k, *values, **kwargs):
    result = original_topk(tensor, k, *values, **kwargs)
    if active and k == 7 and tensor.ndim == 2 and tensor.shape[-1] > 100000:
        if recording:
            captured['top'].append(result.indices.detach().clone())
        else:
            expected = program['top'][indices['top']]
            assert expected.shape == result.indices.shape
            result = torch.return_types.topk((result.values, expected))
        indices['top'] += 1
    return result


def sample(module, logits, *values, **kwargs):
    global profiler
    result = original_sample(module, logits, *values, **kwargs)
    if active:
        if os.environ.get('MOSS_MTP_LOGITS') == '1':
            captured['logits'].append(logits.detach().clone())
        if recording:
            captured['sample'].append(result.detach().clone())
        else:
            expected = program['sample'][indices['sample']]
            assert expected.shape == result.shape
            result = expected
        indices['sample'] += 1
        if os.environ.get('MOSS_MTP_PROFILE') == '1' and generation == 2:
            if indices['sample'] == 8:
                torch.hpu.synchronize()
                profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU])
                profiler.__enter__()
            if indices['sample'] == 10:
                torch.hpu.synchronize()
                profiler.__exit__(None, None, None)
                profiler.profiler.export_chrome_trace(str(root/'mtp-steps.trace.json'))
                (root/'mtp-steps.cpu.txt').write_text(profiler.key_averages().table(sort_by='self_cpu_time_total', row_limit=60))
    return result


def widths(*values, **kwargs):
    value = original_widths(*values, **kwargs)
    if active:
        if recording:
            captured['widths'].append(value)
        else:
            value = tuple(program['widths'][indices['widths']])
        indices['widths'] += 1
    return value


def prepare(cache, num_tokens):
    result = original_prepare(cache, num_tokens)
    tracker = cache.ctl_tracker
    key = tracker['step']
    if active and key not in control_seen:
        control_seen.add(key)
        row = tracker['host'][tracker['parity']].tolist()
        record = [list(key), row]
        if recording:
            captured['controls'].append(record)
        else:
            assert record == program['controls'][indices['controls']], (record, indices['controls'])
        indices['controls'] += 1
    return result


def generate(clip, *values, **kwargs):
    global generation, active, indices, captured, control_seen
    generation += 1
    active = not recording or generation == 2
    indices = dict(top=0, sample=0, widths=0, controls=0)
    captured = dict(top=[], sample=[], widths=[], controls=[], logits=[])
    control_seen = set()
    result = original_generate(clip, *values, **kwargs)
    if active:
        if recording:
            captured['top'] = [x.cpu() for x in captured['top']]
            captured['sample'] = [x.cpu() for x in captured['sample']]
            captured['ids'] = result
            captured.pop('logits')
            torch.save(captured, program_path)
        else:
            assert result == program['ids']
            assert all(indices[k] == len(program[k]) for k in indices), indices
        if os.environ.get('MOSS_MTP_LOGITS') == '1' and not recording:
            torch.save(torch.cat([x.cpu() for x in captured['logits']]), root / f'mtp-logits-{generation}.pt')
        with (root / 'program-checks.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(generation=generation, counts=indices, tokens=len(result)))+'\n')
    active = False
    return result


torch.Tensor.topk = topk
BaseGenerate.sample_token = sample
qwen35.best_widths = widths
qwen35.LinearKV.prepare = prepare
comfy.sd.CLIP.generate = generate
if os.environ.get('MOSS_STATIC_CONTROL'):
    import moss_static_control
    moss_static_control.install(os.environ['MOSS_STATIC_CONTROL'])
if os.environ.get('MOSS_DECODE_REGIONS'):
    import moss_decode_regions
    moss_decode_regions.install()
if os.environ.get('MOSS_NORM_COMPILE') == '1':
    from comfy.text_encoders.llama import RMSNorm
    RMSNorm.forward = torch.compile(RMSNorm.forward, backend='hpu_backend', dynamic=None, options={'force_static_compile': True})
if os.environ.get('MOSS_FINAL_REGION'):
    import moss_text_final_regions
    moss_text_final_regions.install(os.environ['MOSS_FINAL_REGION'])
if os.environ.get('MOSS_LINEAR_REGION'):
    import moss_linear_regions
    moss_linear_regions.install(os.environ['MOSS_LINEAR_REGION'])
if os.environ.get('MOSS_PENDING_REPLAY'):
    import moss_pending_replay
    moss_pending_replay.install(os.environ['MOSS_PENDING_REPLAY'])
if os.environ.get('MOSS_TREE_DELTA') == '1':
    import moss_tree_delta_install
    moss_tree_delta_install.install()
if os.environ.get('MOSS_CONV_CONTROL') == '1':
    import moss_conv_control
    moss_conv_control.install()
if os.environ.get('MOSS_DELTA_CAPTURE') == '1':
    import moss_delta_capture
    moss_delta_capture.install()
if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('moss_gdn_experiment.py')), run_name='__main__')
