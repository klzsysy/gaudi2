"""Investigation-only launcher. Run in the independent product checkout."""
import contextvars
import json
import os
from pathlib import Path
import runpy
import sys
import time

root = Path.cwd()
sys.path.insert(0, str(root))
sys.argv[0] = "main.py"
import comfy.options
comfy.options.enable_args_parsing()
import execution
import comfy.sd
import comfy.utils
import torch
import habana_frameworks.torch.utils.debug as debug
if os.environ.get('MOSS_HEAD_BASELINE') == '1':
    import moss_head_baseline
    moss_head_baseline.install()

if os.environ.get('MOSS_MTP_FIXED') == '1':
    import moss_mtp_program
elif os.environ.get('MOSS_TREE_BASELINE') == '1':
    import moss_gdn_baseline
    moss_gdn_baseline.install()
if os.environ.get('MOSS_TREE_AB') == '1':
    import moss_gdn_baseline
    import moss_head_baseline
if os.environ.get('MOSS_REGION_BASELINE') == '1':
    import moss_gdn_baseline
    moss_gdn_baseline.disable_region()

if os.environ.get("MOSS_KV_MODE"):
    from moss_kv_candidate import install
    install(os.environ["MOSS_KV_MODE"])
if os.environ.get("MOSS_DISABLE_ATTENTION_REGION") == "1":
    from comfy.text_encoders.qwen35 import GatedAttention
    GatedAttention.forward = GatedAttention._forward
if os.environ.get("MOSS_DECODE_REGIONS"):
    import moss_decode_regions
    moss_decode_regions.install()
if os.environ.get("MOSS_BF16_CAPTURE"):
    import moss_bf16_capture
if os.environ.get("MOSS_BF16_CANDIDATE"):
    import moss_bf16_candidates
    moss_bf16_candidates.install(os.environ["MOSS_BF16_CANDIDATE"])

out = Path(os.environ["MOSS_EVENTS"])
diagnostic = False
recompute = True
recompute_types = {"TextGenerate", "CLIPTextEncode", "KSampler", "VAEDecode"}
detail = False
if os.environ.get('MOSS_NORM_EAGER') == '1':
    from comfy.text_encoders import qwen35
    qwen35.RMSNorm.forward = qwen35.LlamaRMSNorm.forward
if os.environ.get('MOSS_NORM_AB') == '1':
    import moss_norm_ab
if os.environ.get('MOSS_PREFILL_AB') == '1':
    from comfy.text_encoders import hpu_gated_delta
    prefill_compiler = hpu_gated_delta.compiled_prefill
    prefill_baseline = False
    hpu_gated_delta.compiled_prefill = lambda function: function if prefill_baseline else prefill_compiler(function)
if os.environ.get('MOSS_ROPE_AB') == '1':
    from comfy.ldm.moss_image import hpu as moss_hpu
    blockwise_coefficients = moss_hpu.blockwise_coefficients
    rope_baseline = False
    moss_hpu.blockwise_coefficients = lambda pe: False if rope_baseline else blockwise_coefficients(pe)
current = contextvars.ContextVar("stage", default={})


def event(kind, **fields):
    with out.open("a") as stream:
        stream.write(json.dumps({"kind": kind, "wall_ns": time.time_ns(),
                                 "mono_ns": time.perf_counter_ns(), **current.get(), **fields}) + "\n")


# Keep loader results/model objects; force the actual computation on every request.
changed = execution.IsChangedCache.get
async def get_changed(self, node_id):
    if recompute and self.dynprompt.get_node(node_id)["class_type"] in recompute_types:
        return self.prompt_id
    return await changed(self, node_id)
execution.IsChangedCache.get = get_changed

execute = execution.PromptExecutor.execute_async
async def execute_request(self, prompt, prompt_id, *args, **kwargs):
    global diagnostic, recompute, detail, prefill_baseline, recompute_types, rope_baseline, norm_baseline
    extra = args[0] if args else kwargs.get("extra_data", {})
    diagnostic = extra.get("moss_diagnostic", False)
    recompute = extra.get("moss_recompute", True)
    recompute_types = set(extra.get('moss_recompute_nodes', ("TextGenerate", "CLIPTextEncode", "KSampler", "VAEDecode")))
    detail = extra.get("moss_breakdown", False)
    prefill_baseline = extra.get('moss_prefill_baseline', False)
    rope_baseline = extra.get('moss_rope_baseline', False)
    norm_baseline = extra.get('moss_norm_baseline', False)
    if os.environ.get('MOSS_TREE_AB') == '1':
        moss_gdn_baseline.hpu_gated_delta.compiled_ops = (moss_gdn_baseline.baseline_ops if extra.get('moss_tree_baseline',False)
                                                        else moss_gdn_baseline.product_ops)
        moss_head_baseline.moss._attention_heads = (moss_head_baseline.together if extra.get('moss_tree_baseline',False)
                                                  else moss_head_baseline.product_head)
        if extra.get('moss_tree_baseline',False):
            moss_gdn_baseline.disable_region()
        else:
            moss_gdn_baseline.qwen35._compiled_hpu_delta = moss_gdn_baseline.product_region
    if os.environ.get('MOSS_NORM_AB') == '1' and 'moss_norm_baseline' in extra:
        moss_norm_ab.baseline = norm_baseline
        moss_norm_ab.install()
    token = current.set({"prompt_id": prompt_id})
    event("request_start", diagnostic_sync=diagnostic,
          recomputed_classes=sorted(recompute_types) if recompute else [],
          cache_control="invalidate compute nodes; retain loader outputs" if recompute else "normal node cache",
          detailed_sync=detail)
    try:
        return await execute(self, prompt, prompt_id, *args, **kwargs)
    finally:
        torch.hpu.synchronize()
        event("request_end", fallback_counts=debug._get_fallback_op_count(),
              memory_stats=torch.hpu.memory_stats())
        current.reset(token)
execution.PromptExecutor.execute_async = execute_request

output = execution.get_output_data
async def get_output(prompt_id, unique_id, obj, *args, **kwargs):
    token = current.set({"prompt_id": prompt_id, "node": unique_id, "class_type": type(obj).__name__})
    if diagnostic:
        torch.hpu.synchronize()
    event("node_start")
    try:
        result = await output(prompt_id, unique_id, obj, *args, **kwargs)
        if diagnostic and type(obj).__name__ in {"KSampler", "VAEDecode"}:
            def inspect(value):
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().float()
                    event("tensor_check", shape=list(value.shape), finite=bool(torch.isfinite(value).all()),
                          minimum=float(value.min()), maximum=float(value.max()), std=float(value.std()))
                elif isinstance(value, dict):
                    for child in value.values():
                        inspect(child)
                elif isinstance(value, (list, tuple)):
                    for child in value:
                        inspect(child)
            inspect(result[0])
        return result
    finally:
        if diagnostic:
            torch.hpu.synchronize()
        event("node_end")
        current.reset(token)
execution.get_output_data = get_output

generate = comfy.sd.CLIP.generate
def generate_text(self, tokens, *args, **kwargs):
    event("generate_start", token_shapes={k: len(v[0]) if v and isinstance(v[0], list) else len(v) for k,v in tokens.items()},
          parameters=kwargs, cpu_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
          inference_mode=torch.is_inference_mode_enabled(), grad_enabled=torch.is_grad_enabled())
    ids = generate(self, tokens, *args, **kwargs)
    event("generate_end", token_ids=ids, output_tokens=len(ids), text=self.decode(ids))
    return ids
comfy.sd.CLIP.generate = generate_text

update = comfy.utils.ProgressBar.update_absolute
def update_progress(self, value, total=None, preview=None):
    if diagnostic and current.get().get("class_type") == "KSampler":
        torch.hpu.synchronize()
    event("progress", value=value, total=total if total is not None else self.total)
    return update(self, value, total, preview)
comfy.utils.ProgressBar.update_absolute = update_progress

if os.environ.get("MOSS_PIPELINE_DETAIL") == "1":
    import moss_pipeline_detail
    moss_pipeline_detail.install(event, lambda: detail)
if os.environ.get("MOSS_PREPARATION_AUDIT") == "1":
    import moss_preparation_audit
    moss_preparation_audit.install(event)
if os.environ.get("MOSS_CFG_BATCH") == "1":
    import moss_cfg_batch_probe
    moss_cfg_batch_probe.install()

runpy.run_path(str(root / "main.py"), run_name="__main__")
