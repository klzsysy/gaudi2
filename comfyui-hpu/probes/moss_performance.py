"""Profile MOSS text generation from the independent checkout with server stopped."""
import json
import os
import statistics
import time
import types
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
if os.environ.get("MOSS_NATIVE_QUANT") == "1":
    _mixed_precision_ops = comfy.ops.mixed_precision_ops
    def _native_mixed_precision_ops(quant_config={}, compute_dtype=torch.bfloat16,
                                    full_precision_mm=False, disabled=[]):
        print("NATIVE_QUANT requested", {"requested_full_precision_mm": full_precision_mm}, flush=True)
        return _mixed_precision_ops(quant_config, compute_dtype, full_precision_mm=False, disabled=disabled)
    comfy.ops.mixed_precision_ops = _native_mixed_precision_ops
    print("NATIVE_QUANT enabled: full_precision_mm=False", flush=True)
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


def install_lm_head_fp8_for_probe():
    """Investigation-only BF16 lm_head -> Gaudi fp8_gemm_v2 projection."""
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = float(torch.finfo(torch.float8_e4m3fnuz).max) if hasattr(torch, "float8_e4m3fnuz") else 240.0
    for name, module in clip.cond_stage_model.named_modules():
        if not name.endswith("lm_head") or not hasattr(module, "weight"):
            continue
        weight = module.weight
        if weight.dtype not in (torch.bfloat16, torch.float16):
            print("LM_HEAD_FP8_SKIP", name, "dtype", weight.dtype, flush=True)
            return
        qweight = torch.empty(weight.shape, dtype=fp8_dtype, device=weight.device)
        weight_scale = torch.empty((weight.shape[0],), dtype=torch.float32, device=weight.device)
        step = 16384
        start_time = time.perf_counter()
        for start in range(0, weight.shape[0], step):
            end = min(start + step, weight.shape[0])
            part = weight[start:end]
            part_scale = (part.abs().amax(dim=-1, keepdim=True).float() + 1e-8) / fp8_max
            qweight[start:end] = torch.ops.hpu.cast_to_fp8_v2(
                part, part_scale.reciprocal(), False, False, fp8_dtype
            )[0]
            weight_scale[start:end] = part_scale.squeeze(-1)
        qweight_t = qweight.t().contiguous()

        def fp8_forward(self, input, *args, **kwargs):
            input_shape = input.shape
            input_2d = input.reshape(-1, input_shape[-1])
            input_scale = (input_2d.abs().amax(dim=-1, keepdim=True).float() + 1e-8) / fp8_max
            input_fp8 = torch.ops.hpu.cast_to_fp8_v2(
                input_2d, input_scale.reciprocal(), False, False, fp8_dtype
            )[0]
            output = torch.ops.hpu.fp8_gemm_v2(
                A=input_fp8,
                trans_A=False,
                B=qweight_t,
                trans_B=False,
                D=None,
                out_dtype=input.dtype,
                A_scale_inv=input_scale,
                B_scale_inv=weight_scale,
                bias=None,
                accumulate=False,
            )
            return output.reshape(*input_shape[:-1], qweight.shape[0])

        module._moss_original_forward = module.forward
        module.forward = types.MethodType(fp8_forward, module)
        torch.hpu.synchronize()
        print("LM_HEAD_FP8", name, {"seconds": time.perf_counter() - start_time,
                                    "weight_shape": list(weight.shape),
                                    "weight_bytes": qweight.numel() * qweight.element_size(),
                                    "scale_bytes": weight_scale.numel() * weight_scale.element_size(),
                                    "fp8_max": fp8_max}, flush=True)
        return
    print("LM_HEAD_FP8_SKIP no lm_head", flush=True)

torch.set_grad_enabled(False)
torch.set_num_threads(16)
mode = os.environ.get("MOSS_PROFILE", "baseline")
mtp_name = os.environ.get("MOSS_MTP", "off")
mtp = True if mtp_name == "on" else (mtp_name if mtp_name in {"2", "3", "4", "5", "tree2s1", "tree2s2", "tree3s1", "tree4s1m8"} else mtp_name == "auto")
result_dir = Path(os.environ.get("MOSS_RESULT_DIR", "test-results"))
result_dir.mkdir(parents=True, exist_ok=True)
clip = comfy.sd.load_clip(
    [os.environ.get("MOSS_CLIP_FILE", "models/text_encoders/qwen3.5_9b_int8_convrot.safetensors")],
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
if os.environ.get("MOSS_LM_HEAD_FP8") == "1":
    install_lm_head_fp8_for_probe()
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


repeat = int(os.environ.get("MOSS_REPEAT", "0"))
if repeat > 0:
    warmup = generate(2)
    records = [generate(32) for _ in range(repeat)]
    seconds = [record["seconds"] for record in records]
    print(json.dumps({
        "repeat": repeat,
        "warmup_seconds": warmup["seconds"],
        "seconds": seconds,
        "median_seconds": statistics.median(seconds),
        "min_seconds": min(seconds),
        "max_seconds": max(seconds),
        "ids_equal": all(record["ids"] == records[0]["ids"] for record in records),
    }), flush=True)
    print("fallback_counts", hpu_debug._get_fallback_op_count(), flush=True)
    raise SystemExit(0)

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
                                with_stack=os.environ.get("MOSS_STACK") == "1",
                                schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1)) as prof:
        result = generate(8)
        prof.step()
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40), flush=True)
    if os.environ.get("MOSS_STACK") == "1":
        print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total", row_limit=80), flush=True)
    prof.profiler.export_chrome_trace(str(result_dir / f"moss-{mode}-trace.json"))
    print("profiler_overhead_ratio", result["seconds"] / control["seconds"], flush=True)
    print("profile_control_ids_equal", result["ids"] == control["ids"], flush=True)
print("fallback_counts", hpu_debug._get_fallback_op_count(), flush=True)
if os.environ.get("MOSS_TRACE_DEQUANT") == "1":
    print("dequant_stats", json.dumps(dequant_stats), flush=True)
(result_dir / f"moss-{mode}-timing.json").write_text(json.dumps(result, indent=2))
