"""Actual Qwen norm inputs versus original operation and FP64 reference."""
import json
import os
from pathlib import Path
import runpy
import torch

state = runpy.run_path(str(Path(__file__).with_name('moss_gdn_experiment.py')), run_name='__main__')
from comfy.text_encoders import qwen35
clip, root = state['clip'], state['directory']
names = {id(m): n for n,m in clip.cond_stage_model.named_modules() if isinstance(m,qwen35.RMSNorm)}
qk_capture = os.environ.get('MOSS_NORM_QK') == '1'
handles = []
if qk_capture:
    for name, module in clip.cond_stage_model.named_modules():
        if isinstance(module, qwen35.RMSNorm) and name.endswith(('q_norm', 'k_norm')):
            handles.append(module.register_forward_pre_hook(lambda module, values: None))
(root/'norm-names.json').write_text(json.dumps(list(names.values()),indent=2))
original = qwen35.RMSNorm.forward
native_function = None
if os.environ.get('MOSS_NORM_KERNEL') == 'native':
    from moss_text_final_regions import native_norm
    native_function = torch.compile(native_norm, backend='hpu_backend', dynamic=None, options={'force_static_compile': True})
seen = set()
records = []

def forward(module, x):
    if torch.compiler.is_compiling():
        return original(module,x)
    name = names.get(id(module),'')
    selected = any(f'layers.{i}.input_layernorm' in name for i in (0,15,31)) or ('mtp' in name and 'layers' not in name)
    if qk_capture:
        selected = name.endswith(('q_norm', 'k_norm')) and (any(f'layers.{i}.' in name for i in (15,31)) or 'mtp' in name)
    key = name, tuple(x.shape), str(x.dtype)
    result = original(module,x)
    if selected and key not in seen:
        seen.add(key)
        eager = qwen35.LlamaRMSNorm.forward(module,x).cpu().double()
        actual = (qwen35._compiled_hpu_rms_norm()(module,x) if qk_capture else result).cpu().double()
        if native_function is not None:
            actual = native_function(module,x).cpu().double()
        values = x.cpu().double()
        scale = module.scale().to(x.dtype).cpu().double()
        reference = values * torch.rsqrt(values.square().mean(-1,keepdim=True)+module.eps) * scale
        denom = reference.square().mean().sqrt()
        base_error = float((eager-reference).square().mean().sqrt()/denom)
        candidate_error = float((actual-reference).square().mean().sqrt()/denom)
        record = dict(name=name,shape=list(x.shape),dtype=str(x.dtype),eager_rrmse=base_error,candidate_rrmse=candidate_error,ratio=candidate_error/base_error if base_error else None)
        records.append(record)
        torch.save(dict(x=x.cpu(),weight=module.weight.cpu(),eps=module.eps,add=module.add),root/f'norm-{len(records)}.pt')
        (root/'norm-reference.json').write_text(json.dumps(records,indent=2))
    return result
qwen35.RMSNorm.forward = forward
clip.generate(state['tokens'],max_length=32,do_sample=True,temperature=.7,top_k=64,top_p=.95,min_p=.05,seed=1,mtp=True)
for text in [Path(os.environ['MOSS_CAPTION']).read_text(),state['workflow']['18:5']['inputs']['text']]:
    clip.encode_from_tokens(clip.tokenize(text))
for handle in handles:
    handle.remove()
print(json.dumps(records,indent=2))
