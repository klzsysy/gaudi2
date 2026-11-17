"""Isolated P1/P2/P3 candidates against the staged four-token implementation."""
import inspect
import textwrap

import torch
import comfy.ops
from comfy.text_encoders import llama, qwen35, hpu_kv_attention


def install(mode):
    baseline_source = inspect.getsource(qwen35.Qwen35._generation_forward)
    if 'def _generation_forward(' not in baseline_source or 'rotary_table' in baseline_source:
        raise RuntimeError('These dated candidates require the recorded readback-only baseline. '
                           'Use its source snapshot or adapt the probe to the current implementation.')
    parts = set(mode.split('+'))
    if 'state2' in parts:
        parts.add('state')
    if 'sharedmlp' in parts:
        parts.add('shared')
    if 'combinedmlp' in parts:
        parts.add('combined')
    if 'shared' in parts or 'combined' in parts:
        original = comfy.ops.CombinedLinear.forward

        def combined(group, x):
            if x.device.type != 'hpu' or x.shape[1] != 1 or not all(
                    getattr(layer, 'layout_type', None) == 'HPUFP8E4M3Layout'
                    and not layer._full_precision_mm and getattr(layer, 'input_scale', None) is None
                    for layer in group.linears):
                return original(group, x)
            flat = x.reshape(-1, x.shape[-1])
            scale = flat.abs().amax(-1, keepdim=True).float().clamp_min(1e-12) / 240.0
            quantized = torch.ops.hpu.cast_to_fp8_v2(flat, scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
            if 'combined' in parts:
                weight, weight_scale = group.hpu_packed
                output = torch.ops.hpu.fp8_gemm_v2(quantized, False, weight, True, None,
                                                 x.dtype, scale, weight_scale, None, False)
            else:
                outputs = []
                for layer in group.linears:
                    weight = layer.weight
                    outputs.append(torch.ops.hpu.fp8_gemm_v2(
                        quantized, False, weight._qdata, True, None, x.dtype,
                        scale, weight._params.scale, layer.bias, False))
                output = torch.cat(outputs, -1)
            return output.reshape(*x.shape[:-1], -1)

        comfy.ops.CombinedLinear.forward = combined
        if parts & {'sharedmlp', 'combinedmlp'}:
            original_mlp = llama.MLP.forward

            def mlp(module, x, norm=None):
                if x.shape[1] != 1 or module.merged_mlp:
                    return original_mlp(module, x, norm)
                if norm is not None:
                    x = norm(x)
                projected = module.combined_gate_up.forward(x)
                if projected is NotImplemented:
                    return original_mlp(module, x)
                gate, up = projected.chunk(2, dim=-1)
                return module.down_proj(module.activation(gate) * up)

            llama.MLP.forward = mlp
    original_factory = qwen35.Qwen35._generation_forward
    source = textwrap.dedent(inspect.getsource(original_factory))
    if 'state-layout' in parts:
        source = source.replace('''                if type(cache) is LinearKV:
''', '''                if type(cache) is LinearKV:
                    cache.value = cache.value.transpose(-1, -2).contiguous().transpose(-1, -2)
''')
    if 'conv' in parts:
        source = source.replace('''                if type(cache) is LinearKV:
''', '''                if type(cache) is LinearKV:
                    cache.key = cache.key.transpose(-1, -2).contiguous().transpose(-1, -2)
''')
    if 'state' in parts:
        source = source.replace('proxies = None', '''proxies = None
    capacity = embeds.shape[1] + max(cache.key.shape[2] for cache in past_key_values if type(cache) is llama.FixedKVCache)
    all_positions = torch.arange(capacity, device=embeds.device).unsqueeze(0)
    all_rotary = rope_matrix(self.model.compute_freqs_cis(all_positions, embeds.device))''')
        source = source.replace('for cache in caches:\n            cache.prepare(1)', '''for cache in caches:
            if type(cache) is not LinearKV:
                cache.prepare(1)''')
        source = source.replace('rotary = rope_matrix(self.model.compute_freqs_cis(positions, current.device))', '''rotary = all_rotary[:, :, caches[0].index:caches[0].index + 1] if kwargs.get("position_ids") is None else rope_matrix(self.model.compute_freqs_cis(positions, current.device))''')
    if 'state2' in parts:
        source = source.replace('embeds.shape[1] + max(cache.key.shape[2]', 'max(cache.key.shape[2]')
        source = source.replace('''        if positions is None:
            positions = torch.arange(caches[0].index, caches[0].index + 1, device=current.device).unsqueeze(0)
''', '')
    if 'control' in parts:
        source = source.replace('for cache in caches:\n            cache.prepare(1)', '''for cache in caches:
            if type(cache) is not LinearKV:
                cache.prepare(1)''')
    if 'dense-rope' in parts:
        source = source.replace('hidden = _compiled_hpu_decode()', 'rotary = rotary.clone()\n        hidden = _compiled_hpu_decode()')
    precise_freq = None
    if 'precise' in parts:
        frequency_source = inspect.getsource(llama.precompute_freqs_cis)
        frequency_source = frequency_source.replace(
            'inv_freq_expanded.float() @ position_ids_expanded.float()',
            'inv_freq_expanded.float() * position_ids_expanded.float()')
        frequency_namespace = dict(vars(llama))
        exec(compile(frequency_source, '<elementwise-frequency>', 'exec'), frequency_namespace)
        precompute = frequency_namespace['precompute_freqs_cis']

        def precise_freq(model, positions, device):
            config = model.config
            return precompute(int(config.head_dim * config.partial_rotary_factor), positions,
                              config.rope_theta, rope_dims=config.mrope_section,
                              interleaved_mrope=True, device=device)

        source = source.replace('self.model.compute_freqs_cis(all_positions, embeds.device)',
                                'precise_freq(self.model, all_positions, embeds.device)')
    buckets = [int(part[6:]) for part in parts if part.startswith('bucket')]
    if buckets:
        bucket = buckets[0]
        source = source.replace('hidden = _compiled_hpu_decode()', '''for cache, proxy in zip(caches, proxies):
            if type(proxy) is HPUSingleTokenKV:
                proxy.visible_capacity = min(cache.key.shape[2], ((cache.index + 1 + BUCKET - 1) // BUCKET) * BUCKET)
        hidden = _compiled_hpu_decode()''')

        def attend(cache, query, key, value):
            limit = cache.visible_capacity
            return hpu_kv_attention.single_decode(query, cache.key[:, :, :limit], cache.value[:, :, :limit], cache.seqlen)

        qwen35.HPUSingleTokenKV.attend = attend
    namespace = dict(vars(qwen35), llama=llama, BUCKET=buckets[0] if buckets else None, precise_freq=precise_freq)
    exec(compile(source, '<qwen80-pe-factory>', 'exec'), namespace)
    factory = namespace['_generation_forward']

    def prepare(self, embeds, caches):
        if 'combined' in parts:
            for module in self.model.modules():
                for group in vars(module).values():
                    if isinstance(group, comfy.ops.CombinedLinear) and all(
                            getattr(layer, 'layout_type', None) == 'HPUFP8E4M3Layout'
                            for layer in group.linears):
                        weights = [layer.weight for layer in group.linears]
                        group.hpu_packed = (torch.cat([w._qdata for w in weights], 0),
                                            torch.cat([w._params.scale.reshape(1).expand(w.shape[0]) for w in weights]).reshape(1, -1))
        return factory(self, embeds, caches)

    qwen35.Qwen35._generation_forward = prepare
    if 'head' in parts:
        original_core = qwen35._hpu_decode

        def core(model, hidden, rotary, caches):
            return model.lm_head(original_core(model, hidden, rotary, caches))

        compiled = torch.compile(core, backend='hpu_backend', fullgraph=True, dynamic=False,
                                 options={'force_static_compile': True})
        qwen35._compiled_hpu_decode = lambda: compiled
        namespace['_compiled_hpu_decode'] = qwen35._compiled_hpu_decode
        original_logits = llama.BaseGenerate.logits

        def logits(self, hidden):
            if hidden.shape[-1] == self.model.config.vocab_size:
                return hidden
            return original_logits(self, hidden)

        llama.BaseGenerate.logits = logits
    if 'gqa' in parts:
        hpu_kv_attention.single_decode = hpu_kv_attention._decode
    if any(part.startswith('tpc') for part in parts):
        import native_gdn_immediate.install
    if 'native-rope' in parts:
        def native_rope(query, key, matrix):
            cosine = torch.cat((matrix[..., 0, 0], matrix[..., 1, 1]), dim=-1)
            sine = torch.cat((matrix[..., 1, 0], -matrix[..., 0, 1]), dim=-1)
            query_out = torch.ops.hpu.rotary_pos_embedding(query.float(), sine, cosine, None, 0, 0)
            key_out = torch.ops.hpu.rotary_pos_embedding(key.float(), sine, cosine, None, 0, 0)
            return query_out.to(query.dtype), key_out.to(key.dtype)

        compiled_rope = torch.compile(native_rope, backend='hpu_backend', fullgraph=True, dynamic=False,
                                      options={'force_static_compile': True, 'decomposition_mode': 'none'})
        hpu_kv_attention._compiled_rope = lambda: compiled_rope
    if 'conv' in parts:
        delta_source = textwrap.dedent(inspect.getsource(qwen35.GatedDeltaNet._decode_single))
        start = delta_source.index('        window = torch.cat')
        end = delta_source.index('    query, key, value = convolved.split', start)
        delta_source = delta_source[:start] + '''        accumulation, updated = torch.ops.hpu.causal_conv1d_update(
            proj, cache.conv_state.transpose(1, 2), weight.reshape(channels, -1).t(), bias,
            activation=False)
        rounded = accumulation[:, 0].float()
        convolved = (rounded / (1.0 + torch.exp(-rounded))).to(x.dtype)
        cache.conv_state.copy_(updated.transpose(1, 2))
''' + delta_source[end:]
        delta_namespace = dict(vars(qwen35))
        exec(compile(delta_source, '<native-causal-conv>', 'exec'), delta_namespace)
        qwen35.GatedDeltaNet._decode_single = delta_namespace['_decode_single']
