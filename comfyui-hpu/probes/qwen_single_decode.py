"""Capture and replay the ordinary BF16 Qwen PE decoder without MTP."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import time

import torch
import habana_frameworks.torch.core as htcore
from comfy.cli_args import args as comfy_args

comfy_args.disable_dynamic_vram = True
comfy_args.disable_pinned_memory = True
comfy_args.disable_async_offload = True
comfy_args.use_pytorch_cross_attention = True
import comfy.sd
import comfy_kitchen
from comfy.text_encoders import qwen35, llama, hpu_mlp, hpu_gated_delta, hpu_kv_attention
from moss_native_profile import NativeProfile


def moved(value, device, memo=None):
    if memo is None:
        memo = {}
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        result = value.detach().to(device).clone()
    elif isinstance(value, dict):
        result = {key: moved(item, "cpu" if key in ("host", "single") else device, memo)
                  for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        result = type(value)(moved(item, device, memo) for item in value)
    elif isinstance(value, llama.FixedKV):
        result = copy.copy(value)
        for key, item in vars(value).items():
            setattr(result, key, moved(item, device, memo))
    else:
        result = value
    memo[id(value)] = result
    return result


def direct_regions():
    qwen35._compiled_hpu_rms_norm = lambda: llama.RMSNorm.forward
    qwen35._compiled_hpu_delta_core = lambda: qwen35.GatedDeltaNet._decode
    qwen35._compiled_hpu_attention = lambda: qwen35._hpu_attention
    hpu_gated_delta.compiled_ops = lambda: (comfy_kitchen.deltanet_conv_step_deferred,
                                           hpu_gated_delta._decode_deferred)
    hpu_mlp._compiled_forward = lambda: hpu_mlp._forward
    hpu_kv_attention.compiled_ops = lambda: (hpu_kv_attention._decode, hpu_kv_attention._tree)
    hpu_kv_attention._compiled_rope = lambda: hpu_kv_attention._rope_forward


def copy_tensors(destination, source):
    if isinstance(destination, torch.Tensor):
        destination.copy_(source.to(destination.device))
    elif isinstance(destination, dict):
        for key in destination:
            copy_tensors(destination[key], source[key])
    elif isinstance(destination, (tuple, list)):
        for target, value in zip(destination, source):
            copy_tensors(target, value)
    elif isinstance(destination, llama.FixedKV):
        for key, value in vars(destination).items():
            copy_tensors(value, getattr(source, key))


def error(actual, reference):
    actual, reference = actual.cpu().float(), reference.cpu().float()
    return dict(exact=torch.equal(actual, reference), max_abs=(actual-reference).abs().max().item(),
                rrmse=((actual-reference).square().mean()/reference.square().mean()).sqrt().item(),
                argmax=actual.argmax(-1).tolist(), reference_argmax=reference.argmax(-1).tolist())


parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("--mode", choices=["capture", "eager", "compile", "lazy", "graph"], default="capture")
parser.add_argument("--input", type=Path)
parser.add_argument("--native", action="store_true")
parser.add_argument("--steps", type=int, default=16)
parser.add_argument("--layers", type=int, default=32)
parser.add_argument("--gdn-single", action="store_true")
parser.add_argument("--start-token", type=int, default=33)
parser.add_argument("--no-snapshot", action="store_true")
args = parser.parse_args()
if args.gdn_single:
    import native_gdn_single.install
args.output.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
torch.set_num_threads(32)
source = json.loads(Path("/home/sonny/gaudi2/comfyui-hpu/workflows/qwen21-latest-2026-11-15-api.json").read_text())
settings = source["459:471"]["inputs"]
clip = comfy.sd.load_clip(["/home/sonny/ComfyUI-Models/models/text_encoders/" + source["459:473"]["inputs"]["clip_name"]],
                         clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
tokens = clip.tokenize(settings["prompt"], system_prompt=source["459:475"]["inputs"]["value"],
                       min_length=1, thinking=settings["thinking"])
clip.load_model(tokens)
target = next(module for module in clip.cond_stage_model.modules() if isinstance(module, qwen35.Qwen35))
assert target.mtp is None
model = target.model

if args.mode == "capture":
    original = model.forward
    counter = 0
    run = 0
    active = None
    window_start = 0

    def forward(*values, **kwargs):
        global counter, active, window_start
        counter += 1
        if run == 2 and counter in (args.start_token, args.start_token + 1):
            before = moved(kwargs, "cpu")
        result = original(*values, **kwargs)
        if run == 2 and counter in (args.start_token, args.start_token + 1):
            record = dict(kwargs=before, hidden=result[0].cpu(), logits=target.logits(result[0]).cpu(),
                          after=moved(result[2], "cpu"))
            torch.save(record, args.output / f"step-{counter}.pt")
        if run in (0, 1) and counter == args.start_token:
            if run == 1 and args.native:
                active = NativeProfile(args.output)
                active.__enter__()
            torch.hpu.synchronize()
            window_start = time.perf_counter()
        if run in (0, 1) and counter == args.start_token + args.steps:
            torch.hpu.synchronize()
            duration = time.perf_counter() - window_start
            if active:
                active.__exit__(None, None, None)
                active = None
            name = "window.json" if run == 1 else "plain-window.json"
            (args.output / name).write_text(json.dumps(dict(seconds=duration, forwards=args.steps,
                                                                     inter_token_intervals=args.steps), indent=2))
        return result

    model.forward = forward
    original_sample = llama.BaseGenerate.sample_token

    def sampled(self, *values, **kwargs):
        result = original_sample(self, *values, **kwargs)
        if counter == max(80, args.start_token + args.steps + 16):
            result = torch.full_like(result, self.model.config.stop_tokens[0])
        return result

    llama.BaseGenerate.sample_token = sampled
    rows = []
    for run in range(2 if args.no_snapshot else 3):
        counter = 0
        torch.hpu.synchronize()
        start = time.perf_counter()
        ids = clip.generate(tokens, do_sample=True, max_length=settings["max_length"], temperature=1., top_k=20, top_p=.95,
                            min_p=0., repetition_penalty=1., presence_penalty=1.5,
                            seed=settings["sampling_mode.seed"], mtp=False)
        torch.hpu.synchronize()
        rows.append(dict(run=run, seconds=time.perf_counter() - start, tokens=ids))
        print({key: value for key, value in rows[-1].items() if key != "tokens"}, flush=True)
    (args.output / "runs.json").write_text(json.dumps(rows, indent=2))
else:
    if args.mode != "eager":
        direct_regions()
    lazy = args.mode in ("lazy", "graph")
    data = torch.load(args.input, weights_only=False, map_location="cpu")
    kwargs = moved(data["kwargs"], "hpu")
    caches = kwargs["past_key_values"]
    original_x = kwargs["embeds"]
    x = original_x.clone()
    positions = kwargs["position_ids"]
    if positions is None:
        positions = torch.tensor([[caches[0].index]], device="hpu")
    rotary = llama.rope_matrix(model.compute_freqs_cis(positions, "hpu"))
    for cache in caches:
        cache.prepare(1)
    initial_caches = moved(caches, "cpu")

    def step():
        hidden = x
        for layer, cache in zip(model.layers[:args.layers], caches):
            hidden, _ = layer(hidden, freqs_cis=rotary, past_key_value=cache)
        hidden = model.norm(hidden)
        return target.logits(hidden)

    execute = torch.compile(step, backend="hpu_backend", fullgraph=True, dynamic=False,
                            options={"force_static_compile": True}) if args.mode == "compile" else step

    def sync():
        if lazy:
            htcore.mark_step()
        torch.hpu.synchronize()

    x.copy_(original_x)
    sync()
    start = time.perf_counter()
    expected = execute()
    sync()
    cold = time.perf_counter() - start
    first = expected.cpu()
    torch.save(first, args.output / "logits.pt")
    state_errors = []
    for index, (actual_cache, reference_cache) in enumerate(zip(caches, data["after"])):
        if isinstance(actual_cache, qwen35.LinearKV):
            actual_state = actual_cache.value.cpu()
            reference_state = reference_cache.value
            state_errors.append(dict(layer=index, exact=torch.equal(actual_state, reference_state),
                                     max_abs=(actual_state-reference_state).abs().max().item(),
                                     rrmse=((actual_state-reference_state).square().mean() /
                                            reference_state.square().mean()).sqrt().item()))
    quality = error(first, data["logits"])
    print(dict(cold=cold, quality=quality), flush=True)
    if args.mode == "graph":
        copy_tensors(caches, initial_caches)
        x.copy_(original_x)
        sync()
        graph = torch.hpu.HPUGraph()
        start = time.perf_counter()
        graph.capture_begin()
        expected = step()
        graph.capture_end()
        sync()
        capture_seconds = time.perf_counter() - start
        execute = lambda: graph.replay(asynchronous=True)
        replay_quality = []
        for source in (args.input, args.input.with_name("step-34.pt")):
            snapshot = torch.load(source, weights_only=False, map_location="cpu")
            current = moved(snapshot["kwargs"], "hpu")
            for cache in current["past_key_values"]:
                cache.prepare(1)
            copy_tensors(caches, current["past_key_values"])
            x.copy_(current["embeds"])
            position = current["position_ids"]
            if position is None:
                position = torch.tensor([[current["past_key_values"][0].index]], device="hpu")
            rotary.copy_(llama.rope_matrix(model.compute_freqs_cis(position, "hpu")))
            sync()
            execute()
            sync()
            replay_quality.append(dict(input=source.name, **error(expected, snapshot["logits"])))
    samples = []
    for _ in range(6):
        sync()
        start = time.perf_counter()
        for _ in range(args.steps):
            x.copy_(original_x)
            if lazy:
                htcore.mark_step()
            execute()
            if args.mode == "lazy":
                htcore.mark_step()
        sync()
        samples.append((time.perf_counter() - start) / args.steps)
        print(samples[-1], flush=True)
    record = dict(mode=args.mode, layers=args.layers, cold_seconds=cold, samples_seconds=samples,
                  median_seconds=statistics.median(samples[1:]), quality=quality, state_errors=state_errors)
    if args.mode == "graph":
        record["capture_seconds"] = capture_seconds
        record["replay_quality"] = replay_quality
    (args.output / "replay.json").write_text(json.dumps(record, indent=2))
