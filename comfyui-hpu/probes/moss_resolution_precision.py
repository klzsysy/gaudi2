"""Replay captured long-sequence blocks against prior attention and FP32 math."""
import json
from pathlib import Path
import sys

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.use_pytorch_cross_attention = True
import comfy.sd
import comfy.ops
import comfy.quant_ops
import comfy.model_management
from comfy.ldm.moss_image import model as moss
from habana_frameworks.torch.utils.debug import _get_fallback_op_count


torch.set_grad_enabled(False)
torch.set_num_threads(32)
root = Path(sys.argv[1])
patcher = comfy.sd.load_diffusion_model("/home/sonny/ComfyUI-Models/models/diffusion_models/Moody-Moss-Mix-V4.1_00001__clean-fp8.safetensors")
comfy.model_management.load_models_gpu([patcher])
model = patcher.model.diffusion_model
shadow = moss.Block(3072, 8192, 24, fused_qkv=True, dtype=torch.float32, device="hpu", operations=comfy.ops.manual_cast)
records = []
for source in sorted((root / "captures").glob("*.pt")):
    data = torch.load(source, weights_only=True)
    block = model.layers[data["layer"]]
    for name, target in shadow.named_parameters():
        value = block.get_parameter(name)
        if isinstance(value, comfy.quant_ops.QuantizedTensor):
            value = value.dequantize()
        target.copy_(value.float())
    x = data["x"].to("hpu")
    pe = data["pe"].to("hpu")
    mod = tuple(value.to("hpu") for value in data["mod"])
    options = {"fp8_attention": False, "_moss_hpu_blockwise_rope": data["blockwise"]}
    reference = shadow(x.float(), tuple(value.float() for value in mod), pe, options).float().cpu()
    baseline = block(x.clone(), mod, pe, options).float().cpu()
    fixed = block(x.clone(), mod, pe, dict(options, fp8_attention=True)).float().cpu()
    record = dict(source=source.name, shape=list(x.shape), finite=bool(fixed.isfinite().all()),
                  change_rrmse=((fixed - baseline).norm() / baseline.norm()).item(),
                  baseline_fp32_rrmse=((baseline - reference).norm() / reference.norm()).item(),
                  fixed_fp32_rrmse=((fixed - reference).norm() / reference.norm()).item(),
                  fallback_counts=_get_fallback_op_count())
    assert record["finite"] and not record["fallback_counts"], record
    records.append(record)
    (root / "precision.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(record), flush=True)
