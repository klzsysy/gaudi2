"""Screen SiTU compute and activation/quantization boundaries on real inputs."""
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
import comfy.ops
import comfy.model_management
import comfy.quant_ops
from comfy.ldm.moss_image import model as moss
import habana_frameworks.torch.utils.debug as debug


root=Path(os.environ['MOSS_OPT_ROOT'])
root.mkdir(parents=True,exist_ok=True)
captures=root.parent/'performance-2026-11-12-fp8-deep/baseline/captures'
torch.set_grad_enabled(False)
torch.set_num_threads(32)
patcher=comfy.sd.load_diffusion_model('/home/sonny/ComfyUI-Models/models/diffusion_models/Moody-Moss-Mix-V4.1_00001__clean-fp8.safetensors')
comfy.model_management.load_models_gpu([patcher])
model=patcher.model.diffusion_model


def fp32_situ(x):
    return comfy.ops._situ_eager(x.float()).to(x.dtype)


def sigmoid_situ(x):
    hidden,gate=x.float().chunk(2,dim=-1)
    return ((8*torch.sigmoid(gate/2)-4)*torch.sigmoid(gate)*(50*torch.sigmoid(hidden*0.08)-25)).to(x.dtype)


def exp_situ(x):
    hidden,gate=x.float().chunk(2,dim=-1)
    half=torch.exp(-gate/2)
    tanh_gate=(1-half)/(1+half)
    sigmoid_gate=1/(1+half.square())
    return (4*tanh_gate*sigmoid_gate*25*torch.tanh(hidden/25)).to(x.dtype)


def tensor_quant(x):
    scale=x.abs().amax(-1,keepdim=True).float().clamp_min(1e-12)/240
    quantized=torch.ops.hpu.cast_to_fp8_v2(x,scale.reciprocal(),False,False,torch.float8_e4m3fn)[0]
    return quantized,scale


def situ_quant(x):
    return tensor_quant(comfy.ops._situ_eager(x))


def situ_fp32_quant(x):
    return tensor_quant(fp32_situ(x))


functions=dict(baseline=comfy.ops._situ_eager,fp32=fp32_situ,sigmoid=sigmoid_situ,exp=exp_situ,
               baseline_quant=situ_quant,fp32_quant=situ_fp32_quant)
compiled={name:torch.compile(function,backend='hpu_backend',dynamic=None,options={'force_static_compile':True})
          for name,function in functions.items()}


def measure(function,x):
    for _ in range(3):output=function(x)
    torch.hpu.synchronize()
    samples=[]
    for _ in range(5):
        start=time.perf_counter_ns()
        for _ in range(20):output=function(x)
        torch.hpu.synchronize()
        samples.append((time.perf_counter_ns()-start)/1e9/20)
    return output,statistics.median(samples)


records=[]
for condition in [307,23]:
    for layer in [0,20,41]:
        data=torch.load(captures/f'input-{condition}-12-{layer}.pt',weights_only=True)
        block=model.layers[layer]
        x,pe=data['x'].to('hpu'),data['pe'].to('hpu')
        mod=tuple(v.to('hpu') for v in data['mod'])
        options=dict(fp8_attention=True,_moss_hpu_blockwise_rope=data['blockwise'])
        attn=moss._compiled_block_ops()[0](block,moss.modulate(x,block.norm1,mod,0,None),pe,options)
        after=moss.residual(x,attn,mod,2,None)
        activation=block.proj['ffn_up_gate'](moss.modulate(after,block.norm2,mod,3,None))
        if condition==307:
            torch.save(activation.cpu(),root/f'situ-input-{layer}.pt')
        reference=comfy.ops._situ_eager(activation.cpu().float())
        baseline=None
        for name,function in compiled.items():
            output,seconds=measure(function,activation)
            if isinstance(output,tuple):
                quantized,scale=output
                output=quantized.cpu().float()*scale.cpu()
            else:output=output.cpu().float()
            if name=='baseline':baseline=output
            record=dict(condition=condition,layer=layer,mode=name,seconds=seconds,
                        baseline_rrmse=float((output-baseline).norm()/baseline.norm()),
                        fp32_rrmse=float((output-reference).norm()/reference.norm()),finite=bool(output.isfinite().all()),
                        fallback=debug._get_fallback_op_count())
            records.append(record)
            (root/'situ-screen.json').write_text(json.dumps(records,indent=2))
            print(json.dumps(record),flush=True)
