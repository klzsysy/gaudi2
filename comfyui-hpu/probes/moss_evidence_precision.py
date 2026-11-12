"""Compare expanded real block inputs against decoded-weight FP32 compute."""
import json
import os
from pathlib import Path

import torch
from comfy.cli_args import args
args.disable_dynamic_vram=True
args.use_pytorch_cross_attention=True
import comfy.sd
import comfy.ops
import comfy.quant_ops
import comfy.model_management
from comfy.ldm.moss_image import model as moss
import habana_frameworks.torch.utils.debug as debug


root=Path(os.environ['MOSS_EVIDENCE_ROOT'])
torch.set_grad_enabled(False)
torch.set_num_threads(32)
patcher=comfy.sd.load_diffusion_model('/home/sonny/ComfyUI-Models/models/diffusion_models/Moody-Moss-Mix-V4.1_00001__clean-fp8.safetensors')
comfy.model_management.load_models_gpu([patcher])
model=patcher.model.diffusion_model
references={}
records=[]
for source in sorted((root/'quality/captures').glob('input-*.pt')):
    data=torch.load(source,weights_only=True)
    layer=data['layer']
    block=model.layers[layer]
    if layer not in references:
        shadow=moss.Block(3072,8192,24,fused_qkv=True,dtype=torch.float32,device='hpu',operations=comfy.ops.manual_cast)
        for name,target in shadow.named_parameters():
            value=block.get_parameter(name)
            if isinstance(value,comfy.quant_ops.QuantizedTensor):
                value=value.dequantize()
            target.copy_(value.float())
        references[layer]=shadow
    x,pe=data['x'].to('hpu'),data['pe'].to('hpu')
    mod=tuple(value.to('hpu') for value in data['mod'])
    options=dict(fp8_attention=True,_moss_hpu_blockwise_rope=data['blockwise'])
    output=block(x.clone(),mod,pe,options).cpu().float()
    reference=references[layer](x.float(),tuple(value.float() for value in mod),pe,options).cpu().float()
    delta=output-reference
    record=dict(source=source.name,layer=layer,shape=list(x.shape),
                rrmse=float(delta.norm()/reference.norm()),max_abs=float(delta.abs().max()),
                reference_rms=float(reference.square().mean().sqrt()),finite=bool(output.isfinite().all()),
                reference_finite=bool(reference.isfinite().all()),fallback=debug._get_fallback_op_count())
    assert record['finite'] and record['reference_finite'] and not record['fallback'],record
    records.append(record)
    (root/'quality/precision.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(record),flush=True)
