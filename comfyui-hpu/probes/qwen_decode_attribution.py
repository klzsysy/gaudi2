"""Measure fixed-state decoder ablations without changing product files."""
import argparse
import collections
import copy
import functools
import json
from pathlib import Path
import statistics
import time

import torch
import habana_frameworks.torch.core
from comfy.cli_args import args as comfy_args

comfy_args.disable_dynamic_vram = True
comfy_args.disable_pinned_memory = True
comfy_args.disable_async_offload = True
comfy_args.use_pytorch_cross_attention = True
import comfy.sd
from comfy.text_encoders import hpu_mlp, llama, qwen35


def move(value, memo=None):
    if memo is None:
        memo = {}
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        result = value.to("hpu").clone()
    elif isinstance(value, dict):
        result = {key: item if key in ("host", "single") else move(item, memo)
                  for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        result = type(value)(move(item, memo) for item in value)
    elif isinstance(value, llama.FixedKV):
        result = copy.copy(value)
        for key, item in vars(value).items():
            setattr(result, key, move(item, memo))
    else:
        result = value
    memo[id(value)] = result
    return result


def clone_tensors(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_tensors(item) for item in value)
    if isinstance(value, dict):
        return {key: clone_tensors(item) for key, item in value.items()}
    return value


parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--iterations", type=int, default=16)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
torch.set_num_threads(32)
workflow = json.loads(Path("/home/sonny/gaudi2/comfyui-hpu/workflows/qwen21-latest-2026-11-15-api.json").read_text())
clip = comfy.sd.load_clip([
    "/home/sonny/ComfyUI-Models/models/text_encoders/" + workflow["459:473"]["inputs"]["clip_name"]
], clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
settings = workflow["459:471"]["inputs"]
tokens = clip.tokenize(settings["prompt"], system_prompt=workflow["459:475"]["inputs"]["value"],
                       min_length=1, thinking=settings["thinking"])
clip.load_model(tokens)
target = next(module for module in clip.cond_stage_model.modules() if isinstance(module, qwen35.Qwen35))
assert target.mtp is None
model = target.model
snapshot = torch.load(args.input, weights_only=False, map_location="cpu")
inputs = move(snapshot["kwargs"])
caches = inputs["past_key_values"]
original_x = inputs["embeds"]
x = original_x.clone()
positions = inputs["position_ids"]
if positions is None:
    positions = torch.tensor([[caches[0].index]], device="hpu")
rotary = llama.rope_matrix(model.compute_freqs_cis(positions, "hpu"))
for cache in caches:
    cache.prepare(1)


def step():
    x.copy_(original_x)
    hidden = x
    for layer, cache in zip(model.layers, caches):
        hidden, _ = layer(hidden, freqs_cis=rotary, past_key_value=cache)
    return target.logits(model.norm(hidden))


def measure(function, groups=5):
    function()
    torch.hpu.synchronize()
    samples = []
    submissions = []
    thread_cpu = []
    for _ in range(groups):
        start = time.perf_counter()
        cpu_start = time.thread_time()
        for _ in range(args.iterations):
            function()
        submissions.append((time.perf_counter() - start) / args.iterations)
        thread_cpu.append((time.thread_time() - cpu_start) / args.iterations)
        torch.hpu.synchronize()
        samples.append((time.perf_counter() - start) / args.iterations)
    return dict(median_ms=statistics.median(samples) * 1000,
                samples_ms=[sample * 1000 for sample in samples],
                submit_median_ms=statistics.median(submissions) * 1000,
                main_thread_cpu_median_ms=statistics.median(thread_cpu) * 1000)


actual = step().cpu().float()
reference = snapshot["logits"].float()
report = dict(input=str(args.input), iterations=args.iterations,
              snapshot_logits_exact=torch.equal(actual, reference),
              snapshot_logits_rrmse=((actual - reference).square().mean() / reference.square().mean()).sqrt().item(),
              ablations={}, components={}, raw_gemv={})
report["uninstrumented"] = measure(step)
print("uninstrumented", report["uninstrumented"], flush=True)

mode = "capture"
skip = set()
records = {}


def wrapped(group, key, function):
    @functools.wraps(function)
    def call(*values, **kwargs):
        if mode == "capture":
            saved_values = clone_tensors(values)
            saved_kwargs = clone_tensors(kwargs)
            result = function(*values, **kwargs)
            records[key] = dict(group=group, function=function, values=saved_values,
                                kwargs=saved_kwargs, result=clone_tensors(result))
            return result
        if group in skip:
            # The MLP normally updates x in place. Identity bypass avoids adding
            # an eager clone to the counterfactual critical path.
            if group == "mlp":
                return values[1]
            return records[key]["result"]
        return function(*values, **kwargs)
    return call


original_core = qwen35._compiled_hpu_delta_core()
core_functions = {id(layer.linear_attn): wrapped("gdn_core", f"gdn_core_{i}", original_core)
                  for i, layer in enumerate(model.layers) if hasattr(layer, "linear_attn")}


def routed_core(module, *values, **kwargs):
    return core_functions[id(module)](module, *values, **kwargs)


qwen35._compiled_hpu_delta_core = lambda: routed_core
original_mlp = hpu_mlp.forward
mlp_functions = {id(layer.mlp): wrapped("mlp", f"mlp_{i}", original_mlp)
                 for i, layer in enumerate(model.layers)}


def routed_mlp(module, *values, **kwargs):
    return mlp_functions[id(module)](module, *values, **kwargs)


hpu_mlp.forward = routed_mlp
for index, layer in enumerate(model.layers):
    layer.input_layernorm.forward = wrapped("norm", f"norm_{index}", layer.input_layernorm.forward)
    if hasattr(layer, "linear_attn"):
        for name in ("in_proj_qkv", "in_proj_z", "out_proj"):
            module = getattr(layer.linear_attn, name)
            module.forward = wrapped("gdn_projection", f"gdn_{index}_{name}", module.forward)
    else:
        layer.self_attn.forward = wrapped("attention", f"attention_{index}", layer.self_attn.forward)
model.norm.forward = wrapped("norm", "final_norm", model.norm.forward)
target.logits = wrapped("head", "head", target.logits)
step()
torch.hpu.synchronize()
mode = "measure"
counts = collections.Counter(record["group"] for record in records.values())
report["captured_calls_per_token"] = dict(counts)
print("captured_calls_per_token", dict(counts), flush=True)
assert counts == dict(norm=33, gdn_projection=72, gdn_core=24, mlp=32, attention=8, head=1), counts

groups = ["gdn_core", "mlp", "attention", "gdn_projection", "head", "norm"]
for group in ["baseline"] + groups + ["gdn_core+attention", "all", "baseline_end"]:
    skip = set(groups) if group == "all" else set(group.split("+"))
    report["ablations"][group] = measure(step)
    (args.output / "attribution.json").write_text(json.dumps(report, indent=2))
    print("ablation", group, report["ablations"][group], flush=True)
skip = set()

# Each component sequence rotates through all real layers. The second timing
# reuses buffers, including the in-place MLP activation. Its values evolve,
# while tensor shapes and the executed operations stay fixed.
for group in groups:
    selected = [record for record in records.values() if record["group"] == group]

    def component():
        for record in selected:
            record["function"](*clone_tensors(record["values"]), **record["kwargs"])

    def clones_only():
        for record in selected:
            clone_tensors(record["values"])

    def reuse_buffers():
        for record in selected:
            record["function"](*record["values"], **record["kwargs"])

    report["components"][group] = dict(execute=measure(component), input_clones=measure(clones_only),
                                       reuse_buffers=measure(reuse_buffers))
    (args.output / "attribution.json").write_text(json.dumps(report, indent=2))
    print("component", group, report["components"][group], flush=True)

# Measure the actual dense matrix shapes with the current physical weight
# layouts and round-robin weights. This isolates the library GEMV baseline.
weights = collections.defaultdict(list)
for name, parameter in model.named_parameters():
    if parameter.ndim != 2 or parameter.numel() < 1024**2 or "embed_tokens" in name:
        continue
    if ".mlp." in name:
        group = "mlp"
    elif ".linear_attn." in name:
        group = "gdn_projection"
    elif ".self_attn." in name:
        group = "attention_projection"
    elif "lm_head" in name:
        group = "head"
    else:
        raise ValueError(name)
    vector = torch.ones((1, parameter.shape[1]), dtype=parameter.dtype, device=parameter.device)
    weights[group].append((vector, parameter))
for group, operands in weights.items():
    def gemvs():
        for vector, weight in operands:
            torch.nn.functional.linear(vector, weight)

    weight_bytes = sum(weight.numel() * weight.element_size() for _, weight in operands)
    timing = measure(gemvs)
    report["raw_gemv"][group] = dict(**timing, count=len(operands), weight_bytes=weight_bytes,
        effective_tb_s=weight_bytes / (timing["median_ms"] / 1000) / 1e12,
        shapes=dict(collections.Counter(str(tuple(weight.shape)) for _, weight in operands)))
    (args.output / "attribution.json").write_text(json.dumps(report, indent=2))
    print("raw_gemv", group, report["raw_gemv"][group], flush=True)
report["wrapped_final_baseline"] = measure(step)
(args.output / "attribution.json").write_text(json.dumps(report, indent=2))
