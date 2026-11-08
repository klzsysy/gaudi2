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
if os.environ.get("MOSS_FORCE_DEVICE_EMBED") == "1":
    comfy.ops.vbar_above_watermark = lambda _module: False
    print("FORCE_DEVICE_EMBED enabled", flush=True)

dequant_stats = {"calls": 0, "seconds": 0.0, "by_shape": {}}
if os.environ.get("MOSS_TRACE_DEQUANT") == "1":
    from comfy.quant_ops import QuantizedTensor
    original_dequantize = QuantizedTensor.dequantize

    def traced_dequantize(self, *args, **kwargs):
        start = time.perf_counter()
        result = original_dequantize(self, *args, **kwargs)
        elapsed = time.perf_counter() - start
        dequant_stats["calls"] += 1
        dequant_stats["seconds"] += elapsed
        key = f"{tuple(self.shape)}:{self.dtype}"
        item = dequant_stats["by_shape"].setdefault(key, {"calls": 0, "seconds": 0.0})
        item["calls"] += 1
        item["seconds"] += elapsed
        return result

    QuantizedTensor.dequantize = traced_dequantize

torch.set_grad_enabled(False)
torch.set_num_threads(16)
mode = os.environ.get("MOSS_PROFILE", "baseline")
mtp = os.environ.get("MOSS_MTP", "off") == "auto"
result_dir = Path(os.environ.get("MOSS_RESULT_DIR", "test-results"))
result_dir.mkdir(parents=True, exist_ok=True)
clip = comfy.sd.load_clip(
    ["models/text_encoders/qwen3.5_9b_int8_convrot.safetensors"],
    clip_type=comfy.sd.CLIPType.MOSS_IMAGE,
)
prompt = "A red ceramic teapot on a wooden table beside a window."
system_prompt = "Expand the image description into one paragraph of 50 to 70 English words. Output only the caption."
if os.environ.get("MOSS_WORKFLOW"):
    workflow = json.loads(Path(os.environ["MOSS_WORKFLOW"]).read_text())
    prompt = workflow["18:17"]["inputs"]["value"]
    system_prompt = workflow["18:14"]["inputs"]["value"]
tokens = clip.tokenize(prompt, system_prompt=system_prompt, min_length=1, thinking=False)
clip.load_model(tokens)
torch.hpu.synchronize()
print("loaded", torch.hpu.memory_stats(), flush=True)
print("profiler activities", torch.profiler.supported_activities(), flush=True)
for name, module in clip.cond_stage_model.named_modules():
    if name.endswith(("lm_head", "embed_tokens")):
        print("HEAD", name, type(module), type(module.weight), module.weight.device,
              getattr(module, "_full_precision_mm", None), module.comfy_cast_weights, flush=True)
embed_probe = getattr(clip.cond_stage_model, "model", None)
if embed_probe is not None:
    embed_probe = getattr(embed_probe, "embed_tokens", None)
if embed_probe is not None:
    print("EMBED_PROBE", {"has_v": hasattr(embed_probe, "_v"),
                           "weight_device": str(embed_probe.weight.device),
                           "weight_dtype": str(embed_probe.weight.dtype),
                           "host_rows_capable": hasattr(embed_probe, "host_rows"),
                           "vbar_above_watermark": comfy.ops.vbar_above_watermark(embed_probe)}, flush=True)


def cache_lm_head_for_probe():
    """Investigation-only: hold one BF16 lm_head decode for one process."""
    for name, module in clip.cond_stage_model.named_modules():
        if not name.endswith("lm_head") or not hasattr(module, "weight"):
            continue
        weight = module.weight
        if not hasattr(weight, "dequantize"):
            print("HEAD_CACHE_SKIP", name, "weight is not quantized", flush=True)
            return
        start = time.perf_counter()
        cached = weight.dequantize()
        torch.hpu.synchronize()
        module.weight = torch.nn.Parameter(cached, requires_grad=False)
        print("HEAD_CACHE", name, {"seconds": time.perf_counter() - start,
                                   "dtype": str(cached.dtype), "shape": list(cached.shape),
                                   "bytes": cached.numel() * cached.element_size()}, flush=True)
        return
    print("HEAD_CACHE_SKIP no lm_head", flush=True)

original_mm = torch._int_mm
def traced_mm(a, b):
    print("INT_MM", a.shape, a.device, b.shape, b.device, flush=True)
    traceback.print_stack(limit=15)
    torch._int_mm = original_mm
    return original_mm(a, b)
torch._int_mm = traced_mm

def generate(length):
    start_wall_ns = time.time_ns()
    start = time.perf_counter()
    ids = clip.generate(tokens, max_length=length, do_sample=True, temperature=0.7,
                        top_k=64, top_p=0.95, min_p=0.05,
                        seed=int(os.environ.get("MOSS_SEED", "42")), mtp=mtp)
    torch.hpu.synchronize()
    result = {"seconds": time.perf_counter()-start, "ids": ids, "text": clip.decode(ids),
              "start_wall_ns": start_wall_ns, "end_wall_ns": time.time_ns()}
    print(json.dumps(result), flush=True)
    return result

generate(2)
if os.environ.get("MOSS_CACHE_LM_HEAD") == "1":
    cache_lm_head_for_probe()
if os.environ.get("MOSS_BENCHMARK") == "1":
    result = generate(32)
else:
    control = generate(8)
    (result_dir / f"moss-{mode}-control.json").write_text(json.dumps(control, indent=2))
    activities = [torch.profiler.ProfilerActivity.CPU]
    if os.environ.get("MOSS_HPU_PROFILE") == "1":
        activities.append(torch.profiler.ProfilerActivity.HPU)
    with torch.profiler.profile(activities=activities, record_shapes=True,
                                schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1)) as prof:
        result = generate(8)
        prof.step()
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40), flush=True)
    prof.profiler.export_chrome_trace(str(result_dir / f"moss-{mode}-trace.json"))
    print("profiler_overhead_ratio", result["seconds"] / control["seconds"], flush=True)
    print("profile_control_ids_equal", result["ids"] == control["ids"], flush=True)
print("fallback_counts", hpu_debug._get_fallback_op_count(), flush=True)
if os.environ.get("MOSS_TRACE_DEQUANT") == "1":
    print("dequant_stats", json.dumps(dequant_stats), flush=True)
(result_dir / f"moss-{mode}-timing.json").write_text(json.dumps(result, indent=2))
