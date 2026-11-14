"""Screen an existing Qwen3.5 draft head against the BF16 Qwen Image PE target."""
import argparse
import copy
import gc
import json
from pathlib import Path
import time
import torch
from safetensors import safe_open
from comfy.cli_args import args

args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True
import comfy.ops
import comfy.sd
from comfy.text_encoders import qwen35

p = argparse.ArgumentParser()
p.add_argument('output', type=Path)
p.add_argument('--length', type=int, default=512)
p.add_argument('--runs', type=int, default=4)
p.add_argument('--modes', nargs='+', default=['off','tree3s1','2'])
p.add_argument('--audit-root', action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
base = Path('/home/sonny/gaudi2/comfyui-hpu')
models = Path('/home/sonny/ComfyUI-Models/models/text_encoders')
workflow = json.loads((base/'workflows/qwen21-latest-2026-11-15-api.json').read_text())
settings = workflow['459:471']['inputs']
torch.set_grad_enabled(False)
torch.set_num_threads(32)
clip = comfy.sd.load_clip([str(models/workflow['459:473']['inputs']['clip_name'])], clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
tokens = clip.tokenize(settings['prompt'], system_prompt=workflow['459:475']['inputs']['value'], min_length=1, thinking=settings['thinking'])
clip.load_model(tokens)
target = next(m for m in clip.cond_stage_model.modules() if isinstance(m, qwen35.Qwen35))
assert target.mtp is None
head = qwen35.MTPHead(target.model.config, device='cpu', dtype=torch.bfloat16, ops=comfy.ops.manual_cast)
with safe_open(str(models/'qwen3.5_9b_bf16_from_int8.safetensors'), framework='pt', device='cpu') as source:
    state = {key.removeprefix('mtp.'):source.get_tensor(key) for key in source.keys() if key.startswith('mtp.')}
head.load_state_dict(state, strict=True)
head.to('hpu')
target.mtp = head
del state

counts = {}
audit = False
class AuditComplete(Exception):
    pass
original_forward = target.model.forward
original_draft = head.forward
def forward(*values, **kwargs):
    length = kwargs['embeds'].shape[1]
    key = 'prefill' if length > 8 else f'target_rows_{length}'
    counts[key] = counts.get(key, 0)+1
    if audit and 1 < length <= 8:
        serial = dict(kwargs)
        serial['embeds'] = kwargs['embeds'][:, :1].clone()
        serial['position_ids'] = kwargs['position_ids'][:, :1].clone()
        serial.pop('decode_buffers', None)
        serial['past_key_values'] = copy.deepcopy(kwargs['past_key_values'])
        for cache in serial['past_key_values']:
            tracker = cache.ctl_tracker if isinstance(cache, qwen35.LinearKV) else cache.tracker
            tracker['tree'] = None
        expected_hidden = original_forward(*values, **serial)[0]
        expected = target.logits(expected_hidden).cpu().float()
        actual_hidden = original_forward(*values, **kwargs)[0][:, :1].clone()
        actual = target.logits(actual_hidden).cpu().float()
        report = dict(rows=length, same_root_input=True, exact=torch.equal(expected,actual),
                      max_abs=(actual-expected).abs().max().item(),
                      rrmse=((actual-expected).square().mean()/expected.square().mean()).sqrt().item(),
                      baseline_top1=expected.argmax(-1).item(), candidate_top1=actual.argmax(-1).item())
        torch.save(dict(serial=expected,tree=actual),a.output/'root-logits.pt')
        (a.output/'root-audit.json').write_text(json.dumps(report,indent=2))
        print('root_audit',report,flush=True)
        raise AuditComplete()
    return original_forward(*values, **kwargs)
def draft(*values, **kwargs):
    counts['draft_calls'] = counts.get('draft_calls', 0)+1
    return original_draft(*values, **kwargs)
target.model.forward = forward
head.forward = draft

manifest = dict(target_file=workflow['459:473']['inputs']['clip_name'],
                draft_source='qwen3.5_9b_bf16_from_int8.safetensors', draft_parameters=sum(v.numel() for v in head.parameters()),
                output_limit=a.length, dtype='BF16', note='Process-local draft-head injection. Output quality and normal loader lifecycle are not accepted by this screen.')
(a.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
for mode in a.modes:
    for turn in range(a.runs):
        counts.clear()
        gc.collect()
        torch.hpu.synchronize()
        start = time.perf_counter()
        ids = clip.generate(tokens, do_sample=True, max_length=a.length, temperature=1.0, top_k=20, top_p=.95,
                            min_p=0., repetition_penalty=1., presence_penalty=1.5,
                            seed=settings['sampling_mode.seed'], mtp=False if mode=='off' else mode)
        torch.hpu.synchronize()
        elapsed = time.perf_counter()-start
        target_calls = sum(value for key,value in counts.items() if key.startswith('target_rows'))
        record = dict(mode=mode, turn=turn, cold=turn==0, seconds=elapsed, tokens=len(ids),
                      tokens_per_second=len(ids)/elapsed, counts=dict(counts),
                      output_tokens_per_verify=(len(ids)-1)/target_calls)
        (a.output/f'{mode}-{turn}-tokens.json').write_text(json.dumps(ids))
        (a.output/f'{mode}-{turn}-text.txt').write_text(clip.decode(ids))
        with (a.output/'timings.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
if a.audit_root:
    audit = True
    try:
        clip.generate(tokens, do_sample=True, max_length=a.length, temperature=1.0, top_k=20, top_p=.95,
                      min_p=0., repetition_penalty=1., presence_penalty=1.5,
                      seed=settings['sampling_mode.seed'], mtp='2')
    except AuditComplete:
        pass
