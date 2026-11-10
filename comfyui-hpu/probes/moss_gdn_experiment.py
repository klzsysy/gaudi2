"""Serial MOSS GDN A/B with persisted tokens, timings and HPU telemetry.

Run from the independent product checkout with this directory on PYTHONPATH.
All switches affect this investigation process only.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
from comfy.cli_args import args

args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True

import comfy.sd
import comfy_kitchen
import habana_frameworks.torch.utils.debug as debug

if os.environ.get('MOSS_TEXT_REGION') == 'mlp':
    from comfy.text_encoders.llama import MLP
    MLP.forward = torch.compile(MLP.forward, backend='hpu_backend', fullgraph=True, dynamic=None, options={'force_static_compile': True})

torch.set_grad_enabled(False)
torch.set_num_threads(16)
directory = Path(os.environ['MOSS_RESULT_DIR'])
directory.mkdir(parents=True, exist_ok=True)
mode = os.environ.get('MOSS_GDN_MODE', 'original')
args.disable_comfy_compiler = mode != 'product'
diagnostic = os.environ.get('MOSS_DIAGNOSTIC') == '1'
capture = os.environ.get('MOSS_CAPTURE') == '1'
stages = {}
seen = set()


def wrap(name):
    original = getattr(comfy_kitchen, name)
    compile_this = mode == 'compile' or (mode == 'decode' and name == 'gated_delta_decode_deferred') or (mode == 'conv' and name == 'deltanet_conv_step_deferred')
    function = torch.compile(original, backend='hpu_backend', fullgraph=True, dynamic=False) if compile_this else original

    @functools.wraps(original)
    def measured(*values, **kwargs):
        key = (name, tuple(values[0].shape))
        if capture and key not in seen:
            seen.add(key)
            saved = [x.detach().cpu().clone() if isinstance(x, torch.Tensor) else x for x in values]
            torch.save({'name': name, 'args': saved, 'kwargs': kwargs}, directory / (name + '-' + str(values[0].shape[1]) + '.pt'))
        if not diagnostic:
            return function(*values, **kwargs)
        torch.hpu.synchronize()
        start = time.perf_counter()
        result = function(*values, **kwargs)
        torch.hpu.synchronize()
        entry = stages.setdefault(name, {'calls': 0, 'seconds': 0.0})
        entry['calls'] += 1
        entry['seconds'] += time.perf_counter() - start
        return result
    setattr(comfy_kitchen, name, measured)


for name in ('deltanet_conv_step_deferred', 'gated_delta_decode_deferred'):
    wrap(name)
workflow = json.loads(Path(os.environ['MOSS_WORKFLOW']).read_text())
clip_file = os.environ.get('MOSS_CLIP_FILE', 'models/text_encoders/qwen3.5_9b_bf16_from_int8.safetensors')
started = time.perf_counter()
clip = comfy.sd.load_clip([clip_file], clip_type=comfy.sd.CLIPType.MOSS_IMAGE)
tokens = clip.tokenize(workflow['18:17']['inputs']['value'], system_prompt=workflow['18:14']['inputs']['value'], min_length=1, thinking=False)
clip.load_model(tokens)
if os.environ.get('MOSS_DISABLE_DENSE_MLP') == '1':
    for module in clip.cond_stage_model.modules():
        if hasattr(module, 'hpu_compile_mlp'):
            module.hpu_compile_mlp = False
torch.hpu.synchronize()
metadata = {'mode': mode, 'diagnostic': diagnostic, 'capture': capture, 'model': clip_file,
            'text_region': os.environ.get('MOSS_TEXT_REGION', 'product'),
            'kv_mode': os.environ.get('MOSS_KV_MODE', 'product'), 'decode_regions': os.environ.get('MOSS_DECODE_REGIONS', ''),
            'disable_attention_region': os.environ.get('MOSS_DISABLE_ATTENTION_REGION') == '1',
            'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('comfy/text_encoders').glob('*.py') if p.name in {'llama.py', 'qwen35.py', 'hpu_kv_attention.py', 'hpu_mlp.py', 'hpu_gated_delta.py'}},
            'mlp_compile_modules': sum(bool(getattr(module, 'hpu_compile_mlp', False)) for module in clip.cond_stage_model.modules()),
            'load_seconds': time.perf_counter() - started, 'torch': torch.__version__,
            'memory_loaded': torch.hpu.memory_stats()}
(directory / 'metadata.json').write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata), flush=True)
length = int(os.environ.get('MOSS_LENGTH', '32'))
count = int(os.environ.get('MOSS_REPEATS', '5'))
teacher_path = os.environ.get('MOSS_TEACHER')
teacher_ids = json.loads(Path(teacher_path).read_text().splitlines()[1])['ids'] if teacher_path else None
teacher_logits = []
teacher_index = 0
if teacher_ids is not None:
    from comfy.text_encoders.llama import BaseGenerate
    def teacher_sample(self, logits, *values, **kwargs):
        global teacher_index
        teacher_logits.append(logits.detach().cpu())
        result = torch.full((logits.shape[0], 1), teacher_ids[teacher_index], dtype=torch.long, device=logits.device)
        teacher_index += 1
        return result
    BaseGenerate.sample_token = teacher_sample
records = []
telemetry_file = (directory / 'hpu.csv').open('w')
telemetry = subprocess.Popen(['hl-smi', '--query-aip=timestamp,utilization.aip,utilization.memory,memory.used,power.draw', '--format=csv,nounits', '-l', '100ms'], stdout=telemetry_file)
try:
    for index in range(count + 1):
        if index == 1 and os.environ.get('MOSS_RELOAD_AFTER_COLD') == '1':
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            torch.hpu.synchronize()
            (directory / 'unloaded-memory.json').write_text(json.dumps(torch.hpu.memory_stats(), indent=2))
            clip.load_model(tokens)
        stages.clear()
        teacher_index = 0
        teacher_logits.clear()
        start_ns = time.time_ns()
        start = time.perf_counter()
        mtp_mode = os.environ.get('MOSS_MTP', 'auto')
        mtp = False if mtp_mode == 'off' else True if mtp_mode == 'auto' else mtp_mode
        ids = clip.generate(tokens, max_length=length, do_sample=True, temperature=0.7, top_k=64, top_p=0.95, min_p=0.05, seed=0, mtp=mtp)
        torch.hpu.synchronize()
        record = {'index': index, 'warm': index > 0, 'seconds': time.perf_counter() - start,
                  'start_ns': start_ns, 'end_ns': time.time_ns(), 'ids': ids, 'text': clip.decode(ids),
                  'stages': dict(stages), 'fallback': debug._get_fallback_op_count(),
                  'memory': torch.hpu.memory_stats()}
        records.append(record)
        if teacher_ids is not None:
            torch.save(torch.cat(teacher_logits), directory / f'logits-{index}.pt')
        with (directory / 'runs.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps({k:v for k,v in record.items() if k not in ('ids','text','memory')}), flush=True)
finally:
    telemetry.terminate()
    telemetry.wait(timeout=10)
    telemetry_file.close()
warm = [r['seconds'] for r in records if r['warm']]
summary = {'mode': mode, 'median': statistics.median(warm) if warm else None,
           'seconds': warm, 'ids_equal': all(r['ids'] == records[0]['ids'] for r in records)}
(directory / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
