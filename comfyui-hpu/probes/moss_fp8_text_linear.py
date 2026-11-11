"""Check native text Linear dtype, routing and real weight latency in isolation."""
import json
from pathlib import Path
import time
import torch
from safetensors import safe_open
import comfy.ops as ops
from habana_frameworks.torch.utils.debug import _get_fallback_op_count

torch.set_grad_enabled(False)
torch.set_num_threads(32)
root=Path('/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-11-fp8-system')
path='/home/sonny/ComfyUI-Models/models/text_encoders/qwen3.5_9b_fp8_from_int8.safetensors'
results=[]
with safe_open(path,framework='pt') as f:
 for name in ['model.language_model.layers.0.mlp.up_proj','model.language_model.layers.0.linear_attn.in_proj_qkv','lm_head']:
  w=f.get_tensor(name+'.weight');s=f.get_tensor(name+'.weight_scale')
  factory=ops.mixed_precision_ops(compute_dtype=torch.float32,load_device=torch.device('hpu'))
  layer=factory.Linear(w.shape[1],w.shape[0],bias=False,device='cpu')
  layer.load_state_dict({'weight':w,'weight_scale':s,'comfy_quant':torch.tensor(list(json.dumps({'format':'float8_e4m3fn'}).encode()),dtype=torch.uint8)},strict=False)
  layer.to('hpu')
  print(name,layer.layout_type,layer._full_precision_mm,layer.weight.dtype,flush=True)
  for rows in (1,4,8):
   x=torch.randn(1,rows,w.shape[1],device='hpu',dtype=torch.bfloat16)
   for compiled in (False,True):
    fn=torch.compile(layer,backend='hpu_backend',dynamic=False) if compiled else layer
    for _ in range(2):y=fn(x)
    torch.hpu.synchronize();t=time.perf_counter()
    for _ in range(10):y=fn(x)
    torch.hpu.synchronize()
    r=dict(name=name,rows=rows,compiled=compiled,ms=(time.perf_counter()-t)*100,finite=bool(y.isfinite().all().item()))
    results.append(r);print(r,flush=True)
  del layer
(root/'text-linear.json').write_text(json.dumps(dict(results=results,fallback=_get_fallback_op_count()),indent=2))
