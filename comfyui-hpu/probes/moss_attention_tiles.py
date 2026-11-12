"""Test QK/softmax/AV query tiles inside the existing FP8 block compilation."""
import json
import os
from pathlib import Path
import statistics
import time

import torch
from comfy.cli_args import args
args.disable_dynamic_vram=True
args.use_pytorch_cross_attention=True
import comfy.sd
import comfy.model_management
from comfy.ldm.modules import attention
import habana_frameworks.torch.utils.debug as debug


root=Path(os.environ['MOSS_OPT_ROOT'])
torch.set_grad_enabled(False)
torch.set_num_threads(32)
patcher=comfy.sd.load_diffusion_model('/home/sonny/ComfyUI-Models/models/diffusion_models/Moody-Moss-Mix-V4.1_00001__clean-fp8.safetensors')
comfy.model_management.load_models_gpu([patcher])
model=patcher.model.diffusion_model
original=attention._hpu_fp8_attention_av


def tiled_attention(chunk):
    def forward(q,k,v):
        query,key,value=q[0,0],k[0,0],v[0,0]
        value_scale=value.abs().amax().float().clamp_min(1e-12)/240
        quantized=torch.ops.hpu.cast_to_fp8_v2(value,value_scale.reciprocal(),False,False,torch.float8_e4m3fn)[0]
        factor=q.shape[-1]**-.25
        query=query*factor
        key=(key*factor).transpose(-1,-2)
        probability_scale=torch.full((),1/128,device=q.device,dtype=torch.float32)
        tiles=[]
        for start in range(0,query.shape[0],chunk):
            scores=query[start:start+chunk]@key
            probabilities=torch.ops.hpu.softmax_fp8.Scalar_scales(scores,-1,1.0,128.0)
            output=torch.ops.hpu.fp8_gemm_v2(probabilities,False,quantized,False,None,q.dtype,
                                             probability_scale,value_scale,None,False)
            tiles.append(output)
        return torch.cat(tiles,dim=0).unsqueeze(0).unsqueeze(0)
    return forward


records=[]
for condition in [307,23]:
    data=torch.load(root.parent/f'performance-2026-11-12-fp8-deep/baseline/captures/input-{condition}-12-0.pt',weights_only=True)
    x,pe=data['x'].to('hpu'),data['pe'].to('hpu')
    mod=tuple(value.to('hpu') for value in data['mod'])
    options=dict(fp8_attention=True,_moss_hpu_blockwise_rope=data['blockwise'])
    block=model.layers[0]
    reference=None
    for chunk in [0,256,512,1024,1536,2048]:
        torch._dynamo.reset()
        attention._hpu_fp8_attention_av=tiled_attention(chunk) if chunk else original
        for _ in range(3):output=block(x.clone(),mod,pe,options)
        torch.hpu.synchronize()
        samples=[]
        for _ in range(5):
            start=time.perf_counter_ns()
            for _ in range(5):output=block(x.clone(),mod,pe,options)
            torch.hpu.synchronize()
            samples.append((time.perf_counter_ns()-start)/1e9/5)
        value=output.cpu().float()
        if reference is None:reference=value
        record=dict(condition=condition,chunk=chunk,seconds=statistics.median(samples),
                    rrmse=float((value-reference).norm()/reference.norm()),finite=bool(value.isfinite().all()),
                    fallback=debug._get_fallback_op_count())
        records.append(record)
        (root/'attention-tiles.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(record),flush=True)
