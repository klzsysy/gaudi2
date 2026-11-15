"""Isolated no-MTP Qwen execution candidates for end-to-end validation."""
import copy
import functools

import torch
import torch.nn.functional as F
import comfy.ops
import comfy_kitchen
from comfy.text_encoders import hpu_gated_delta, hpu_kv_attention, hpu_mlp, llama, qwen35
from habana_frameworks.torch.hpex.kernels import FusedSDPA


def immediate_delta(module, x, proj, z, cache):
    """Single-token recurrence with immediately committed conv and FP32 state."""
    batch, _, channels = proj.shape
    heads = module.num_value_heads
    width = module.key_head_dim
    groups = heads // module.num_key_heads
    with comfy.ops.CastBiasWeightContext(module.conv1d, proj, offloadable=True) as (weight, bias):
        window = torch.cat((cache.conv_state, proj.transpose(1, 2)), dim=-1)
        accumulation = (window.float().flip(-1) * weight.reshape(channels, -1).float().flip(-1)).sum(-1)
        if bias is not None:
            accumulation = accumulation + bias
        rounded = accumulation.to(x.dtype).float()
        convolved = (rounded / (1.0 + torch.exp(-rounded))).to(x.dtype)
        cache.conv_state.copy_(window[..., 1:])
    query, key, value = convolved.split((module.key_dim, module.key_dim, module.value_dim), dim=-1)
    query = query.reshape(batch, module.num_key_heads, width).float()
    key = key.reshape(batch, module.num_key_heads, width).float()
    value = value.reshape(batch, heads, module.value_head_dim).float()
    query = query / query.square().sum(-1, keepdim=True).sqrt().clamp(min=1e-12) * width ** -0.5
    key = key / key.square().sum(-1, keepdim=True).sqrt().clamp(min=1e-12)
    query = query.repeat_interleave(groups, dim=1)
    key = key.repeat_interleave(groups, dim=1)
    beta = module.in_proj_b(x).sigmoid().float().reshape(batch, heads, 1)
    decay = torch.exp(cache.g_decay * F.softplus(module.in_proj_a(x).float() + cache.dt_bias)).reshape(batch, heads, 1)
    state = cache.recurrent_state
    prediction = (key.unsqueeze(-1) * state).sum(-2)
    update = beta * (value - decay * prediction)
    output = decay * (query.unsqueeze(-1) * state).sum(-2) + (query * key).sum(-1, keepdim=True) * update
    state.copy_(state * decay.unsqueeze(-1) + key.unsqueeze(-1) * update.unsqueeze(-2))
    output = output.to(x.dtype).float()
    rstd = torch.rsqrt(output.square().mean(-1, keepdim=True) + module.norm.eps)
    normalized = (output * rstd * cache.norm_weight.float()).to(x.dtype)
    return (normalized * F.silu(z.reshape(batch, heads, module.value_head_dim))).unsqueeze(1)


def sdpa_decode(q, key, value, length):
    visible = torch.arange(key.shape[2], device=q.device).view(1, 1, 1, -1) < length.view(-1, 1, 1, 1)
    mask = torch.zeros(visible.shape, device=q.device, dtype=q.dtype).masked_fill(~visible, float("-inf"))
    output = FusedSDPA.apply(q, key, value, mask, 0.0, False, None, "fp32", False)
    return output.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def direct_regions():
    entries = [
        (qwen35, "_compiled_hpu_rms_norm", llama.RMSNorm.forward),
        (hpu_gated_delta, "compiled_ops", (comfy_kitchen.deltanet_conv_step_deferred,
                                           hpu_gated_delta._decode_deferred)),
        (hpu_mlp, "_compiled_forward", hpu_mlp._forward),
        (hpu_kv_attention, "compiled_ops", (hpu_kv_attention._decode, hpu_kv_attention._tree)),
        (hpu_kv_attention, "_compiled_rope", hpu_kv_attention._rope_forward),
        (qwen35, "_compiled_hpu_attention", qwen35._hpu_attention),
    ]
    for owner, name, direct in entries:
        original = getattr(owner, name)

        def select(original=original, direct=direct):
            return direct if torch.compiler.is_compiling() else original()

        setattr(owner, name, select)


def install_whole():
    original_generate = llama.BaseGenerate.generate

    def core(model, hidden, rotary, caches):
        for layer, cache in zip(model.layers, caches):
            hidden, _ = layer(hidden, freqs_cis=rotary, past_key_value=cache)
        return model.norm(hidden)

    compiled = torch.compile(core, backend="hpu_backend", fullgraph=True, dynamic=False,
                             options={"force_static_compile": True})

    @functools.wraps(original_generate)
    def generate(self, *values, **kwargs):
        if not isinstance(self, qwen35.Qwen35):
            return original_generate(self, *values, **kwargs)
        assert self.mtp is None
        model = self.model
        original_forward = model.forward
        had_override = "forward" in model.__dict__
        proxies = None

        def forward(*forward_values, **forward_kwargs):
            nonlocal proxies
            embeds = forward_kwargs.get("embeds")
            caches = forward_kwargs.get("past_key_values")
            if embeds is None or embeds.shape[1] != 1 or not caches or caches[0].index == 0:
                return original_forward(*forward_values, **forward_kwargs)
            if proxies is None:
                proxies = [copy.copy(cache) for cache in caches]
                for proxy in proxies:
                    proxy.index = 1
                    if isinstance(proxy, llama.FixedKVCache):
                        proxy.tracker = {"tree": None}
            for cache in caches:
                cache.prepare(1)
            positions = forward_kwargs.get("position_ids")
            if positions is None:
                positions = torch.arange(caches[0].index, caches[0].index + 1, device=embeds.device).unsqueeze(0)
            rotary = llama.rope_matrix(model.compute_freqs_cis(positions, embeds.device))
            hidden = compiled(model, embeds.clone(), rotary, proxies)
            for cache in caches:
                cache.advance(1)
            return hidden, None, caches

        model.forward = forward
        try:
            return original_generate(self, *values, **kwargs)
        finally:
            if had_override:
                model.forward = original_forward
            else:
                del model.forward

    llama.BaseGenerate.generate = generate


def install(mode):
    parts = set(mode.split("+"))
    if "gdn_native" in parts:
        import native_gdn_single.install
    delta_core = immediate_delta if "gdn_direct" in parts else qwen35.GatedDeltaNet._decode
    if "gdn_direct" in parts:
        compiled_delta = torch.compile(delta_core, backend="hpu_backend", fullgraph=True, dynamic=False,
                                       options={"force_static_compile": True})
        qwen35._compiled_hpu_delta_core = lambda: compiled_delta
    if "sdpa" in parts:
        hpu_kv_attention._decode = sdpa_decode
        hpu_kv_attention.compiled_ops.cache_clear()
    if "bucket" in parts:
        original_prepare = llama.FixedKVCache.prepare
        original_attend = llama.FixedKVCache.attend

        def prepare(cache, rows):
            original_prepare(cache, rows)
            cache.visible_capacity = min(cache.key.shape[2], ((cache.index + rows + 1023) // 1024) * 1024)

        def attend(cache, q, k, v):
            if q.shape[2] != 1 or cache.tracker["tree"] is not None:
                return original_attend(cache, q, k, v)
            decode, _ = hpu_kv_attention.compiled_ops()
            limit = cache.visible_capacity
            return decode(q, cache.key[:, :, :limit], cache.value[:, :, :limit], cache.seqlen)

        llama.FixedKVCache.prepare = prepare
        llama.FixedKVCache.attend = attend
    if "whole" in parts:
        assert "bucket" not in parts
        direct_regions()
        qwen35._compiled_hpu_delta_core = lambda: delta_core
        install_whole()
