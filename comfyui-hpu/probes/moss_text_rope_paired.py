"""Interleave unchanged and compiled opaque RoPE in one fixed-MTP text process."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True
import comfy.sd
import comfy.model_management
from comfy_kitchen.backends.eager.rope import apply_rope_split_half as eager_rope
from habana_frameworks.torch.utils.debug import _get_fallback_op_count
from moss_text_rope_candidate import apply_rope_split_half as functional_rope


parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('--text', choices=['fp8', 'bf16'], required=True)
parser.add_argument('--implementation', choices=['eager', 'functional'], default='eager')
parser.add_argument('--clone-inputs', action='store_true')
parser.add_argument('--fixed-mode', choices=['baseline', 'candidate'])
options = parser.parse_args()
options.root.mkdir(parents=True, exist_ok=True)
investigation = options.root.parent.parent
os.environ.update(MOSS_RESULT_DIR=str(options.root / 'mtp'), MOSS_MTP_PROFILE='0',
                  MOSS_MTP_PROGRAM=str(investigation / 'performance-2026-11-11-completion/mtp-record-298/program.pt'))
import moss_mtp_program as program

torch.set_grad_enabled(False)
torch.set_num_threads(32)
source = json.loads((investigation / 'performance-2026-11-11-fp8-system/native-release/source.json').read_text())
filename = f'/home/sonny/ComfyUI-Models/models/text_encoders/qwen3.5_9b_{options.text}_from_int8.safetensors'
clip = comfy.sd.load_clip([filename], clip_type=comfy.sd.CLIPType.MOSS_IMAGE)
tokens = clip.tokenize(source['18:17']['inputs']['value'], system_prompt=source['18:14']['inputs']['value'],
                       min_length=1, thinking=False)
clip.load_model(tokens)
function = eager_rope if options.implementation == 'eager' else functional_rope
compiled = torch.compile(function, backend='hpu_backend', dynamic=None,
                         options={'force_static_compile': True, 'decomposition_mode': 'none'})
records = []


def run(mode, label, logits=False):
    function = eager_rope if mode == 'baseline' else compiled
    if mode == 'candidate' and options.clone_inputs:
        def function(query, key, frequencies):
            return compiled(query.clone(), key.clone(), frequencies.clone())
    if not options.fixed_mode or not records:
        torch.library.register_kernel('comfy_kitchen::apply_rope_split_half', 'hpu', function)
    os.environ['MOSS_MTP_LOGITS'] = '1' if logits else '0'
    os.environ['MOSS_MTP_LOGITS_GENERATION'] = str(program.generation + 1)
    torch.hpu.synchronize()
    start = time.perf_counter_ns()
    ids = clip.generate(tokens, max_length=4096, do_sample=True, temperature=.7, top_k=64,
                        top_p=.95, min_p=.05, seed=0, mtp=True)
    torch.hpu.synchronize()
    record = dict(mode=mode, label=label, text=options.text, tokens=len(ids),
                  seconds=(time.perf_counter_ns() - start) / 1e9,
                  fallback=_get_fallback_op_count(), generation=program.generation)
    assert not record['fallback']
    records.append(record)
    (options.root / 'runs.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(record), flush=True)
    if logits:
        original = options.root / 'mtp' / f'mtp-logits-{program.generation}.pt'
        original.rename(options.root / f'{mode}-logits.pt')


selected_modes = (options.fixed_mode,) if options.fixed_mode else ('baseline', 'candidate')
for mode in selected_modes:
    run(mode, 'cold')
for trial in range(5):
    order = selected_modes if options.fixed_mode or trial % 2 == 0 else tuple(reversed(selected_modes))
    for mode in order:
        run(mode, f'warm-{trial}')
for mode in selected_modes:
    run(mode, 'diagnostic', logits=True)
summary = {mode: dict(median=statistics.median(r['seconds'] for r in records
                                            if r['mode'] == mode and r['label'].startswith('warm')))
           for mode in selected_modes}
(options.root / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
program.captured.clear()
comfy.model_management.unload_all_models()
comfy.model_management.soft_empty_cache()
torch.hpu.synchronize()
(options.root / 'unloaded-memory.json').write_text(json.dumps(torch.hpu.memory_stats(), indent=2))
