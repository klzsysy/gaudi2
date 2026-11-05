"""Profile MOSS text generation from the independent checkout with server stopped."""
import json
import os
import time
import traceback
from pathlib import Path

import torch
from comfy.cli_args import args

args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True

import comfy.sd
import habana_frameworks.torch.utils.debug as hpu_debug

torch.set_grad_enabled(False)
torch.set_num_threads(16)
mode = os.environ.get("MOSS_PROFILE", "baseline")
mtp = os.environ.get("MOSS_MTP", "off") == "auto"
clip = comfy.sd.load_clip(
    ["models/text_encoders/qwen3.5_9b_int8_convrot.safetensors"],
    clip_type=comfy.sd.CLIPType.MOSS_IMAGE,
)
tokens = clip.tokenize("A red ceramic teapot on a wooden table beside a window.",
                       system_prompt="Expand the image description into one paragraph of 50 to 70 English words. Output only the caption.",
                       min_length=1, thinking=False)
clip.load_model(tokens)
torch.hpu.synchronize()
print("loaded", torch.hpu.memory_stats(), flush=True)
print("profiler activities", torch.profiler.supported_activities(), flush=True)
for name, module in clip.cond_stage_model.named_modules():
    if name.endswith(("lm_head", "embed_tokens")):
        print("HEAD", name, type(module), type(module.weight), module.weight.device,
              getattr(module, "_full_precision_mm", None), module.comfy_cast_weights, flush=True)

original_mm = torch._int_mm
def traced_mm(a, b):
    print("INT_MM", a.shape, a.device, b.shape, b.device, flush=True)
    traceback.print_stack(limit=15)
    torch._int_mm = original_mm
    return original_mm(a, b)
torch._int_mm = traced_mm

def generate(length):
    start = time.perf_counter()
    ids = clip.generate(tokens, max_length=length, do_sample=True, temperature=0.7,
                        top_k=64, top_p=0.95, min_p=0.05, seed=42, mtp=mtp)
    torch.hpu.synchronize()
    result = {"seconds": time.perf_counter()-start, "ids": ids, "text": clip.decode(ids)}
    print(json.dumps(result), flush=True)
    return result

generate(2)
if os.environ.get("MOSS_BENCHMARK") == "1":
    result = generate(32)
else:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
        result = generate(8)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40), flush=True)
    prof.profiler.export_chrome_trace(f"test-results/moss-{mode}-trace.json")
print("fallback_counts", hpu_debug._get_fallback_op_count(), flush=True)
Path(f"test-results/moss-{mode}-timing.json").write_text(json.dumps(result, indent=2))
