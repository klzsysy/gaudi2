"""Process-local experiments for the November 15 Qwen investigation."""
import inspect
import importlib.util
import sys
import textwrap
import copy
import json
import os
from pathlib import Path
import torch
import comfy.ops
from comfy.text_encoders import qwen35, hpu_gated_delta, llama
from comfy.ldm.qwen_image21 import model as dit


def materialized_delta(module, x, kv):
    proj = module.in_proj_qkv(x)
    return materialized_tail(module, x, kv, proj)


def materialized_tail(module, x, kv, proj):
    z = module.in_proj_z(x)
    conv, delta = hpu_gated_delta.compiled_ops()
    with comfy.ops.CastBiasWeightContext(module.conv1d, proj, offloadable=True) as (weight, bias):
        conv(proj, kv.conv_state, weight, bias, kv.proj_buf, kv.qkv_buf, kv.ctl)
    with comfy.ops.CastBiasWeightContext(module.in_proj_a, x, offloadable=True) as (wa, _), \
         comfy.ops.CastBiasWeightContext(module.in_proj_b, x, offloadable=True) as (wb, _):
        core = delta(x, wa, wb, kv.dt_bias, kv.g_decay, kv.recurrent_state,
                     module.key_dim, module.num_key_heads, module.key_head_dim ** -0.5,
                     z, kv.norm_weight, module.norm.eps, kv.qkv_buf, kv.gates_buf, kv.sumsq_buf, kv.ctl)
    output = module.out_proj(core.reshape(x.shape[0], x.shape[1], -1))
    return output, proj, z, core


def decode_tail(module, x, proj, z, kv):
    core = decode_core(module, x, proj, z, kv)
    return module.out_proj(core.reshape(x.shape[0], x.shape[1], -1))


def decode_core(module, x, proj, z, kv):
    conv, delta = hpu_gated_delta.compiled_ops()
    with comfy.ops.CastBiasWeightContext(module.conv1d, proj, offloadable=True) as (weight, bias):
        conv(proj, kv.conv_state, weight, bias, kv.proj_buf, kv.qkv_buf, kv.ctl)
    with comfy.ops.CastBiasWeightContext(module.in_proj_a, x, offloadable=True) as (wa, _), \
         comfy.ops.CastBiasWeightContext(module.in_proj_b, x, offloadable=True) as (wb, _):
        core = delta(x, wa, wb, kv.dt_bias, kv.g_decay, kv.recurrent_state,
                     module.key_dim, module.num_key_heads, module.key_head_dim ** -0.5,
                     z, kv.norm_weight, module.norm.eps, kv.qkv_buf, kv.gates_buf, kv.sumsq_buf, kv.ctl)
    return core


def mlp_residual(module, x, norm):
    output = module(x, norm=norm)
    torch.add(x, output, out=x)
    return x, output


def install(mode):
    parts = mode.split('+')
    if 'rope_cache' in parts:
        source = textwrap.dedent(inspect.getsource(llama.BaseGenerate.generate))
        source = source.replace('if decode_buffers is not None and next_pos is None:',
                                'if (decode_buffers is not None or device.type == "hpu") and next_pos is None:')
        source = source.replace('                if rope is not None and step > 0:\n                    past_len = self.model.get_past_len(past_key_values)\n                    extra["freqs_cis"] = rope[:, :, past_len:past_len + 1]',
                                '            if rope is not None and step > 0:\n                past_len = self.model.get_past_len(past_key_values)\n                extra["freqs_cis"] = rope[:, :, past_len:past_len + 1]')
        namespace = dict(vars(llama))
        exec(compile(source, __file__, 'exec'), namespace)
        llama.BaseGenerate.generate = namespace['generate']
    if 'tail2' in parts or 'core' in parts:
        compiled = torch.compile(decode_core if 'core' in parts else decode_tail, backend='hpu_backend', dynamic=False,
                                 options={'force_static_compile': True})
        original = qwen35.GatedDeltaNet.forward
        def tail_forward(self, x, past_key_value=None, **kwargs):
            if x.device.type == 'hpu' and x.shape[1] == 1 and type(past_key_value) is qwen35.LinearKV and past_key_value.decode_step(1):
                proj, z = self.in_proj_qkv(x), self.in_proj_z(x)
                result = compiled(self, x, proj, z, past_key_value)
                if 'core' in parts:
                    result = self.out_proj(result.reshape(x.shape[0], x.shape[1], -1))
                return result, past_key_value
            return original(self, x, past_key_value=past_key_value, **kwargs)
        qwen35.GatedDeltaNet.forward = tail_forward
    if 'mlpresidual' in parts:
        compiled_mlp = torch.compile(mlp_residual, backend='hpu_backend', dynamic=False,
                                     options={'force_static_compile': True})
        def fused_forward(module, x, norm):
            return compiled_mlp(module, x, norm)[0]
        source = inspect.getsource(qwen35.Qwen35TransformerBlock.forward)
        source = textwrap.dedent(source).replace('mlp_output = hpu_mlp.forward(self.mlp, x, self.post_attention_layernorm)',
                                                'return fused_forward(self.mlp, x, self.post_attention_layernorm), present_key_value')
        namespace = dict(vars(qwen35), fused_forward=fused_forward)
        exec(compile(source, __file__, 'exec'), namespace)
        qwen35.Qwen35TransformerBlock.forward = namespace['forward']
    if 'audit_materialized' in parts or 'audit_tail' in parts:
        compiled = torch.compile(materialized_tail if 'audit_tail' in parts else materialized_delta, backend='hpu_backend', dynamic=False,
                                 options={'force_static_compile': True})
        original = qwen35.GatedDeltaNet.forward
        audited = set()
        def audit(self, x, past_key_value=None, **kwargs):
            if x.shape[1] == 1 and len(audited) < 3 and id(self) not in audited:
                audited.add(id(self))
                copies = []
                for _ in range(2):
                    kv = copy.copy(past_key_value)
                    for key, value in vars(kv).items():
                        if isinstance(value, torch.Tensor):
                            setattr(kv, key, value.clone())
                    copies.append(kv)
                expected = materialized_delta(self, x, copies[0])
                values = (self.in_proj_qkv(x),) if 'audit_tail' in parts else ()
                actual = compiled(self, x, copies[1], *values)
                report = {}
                pairs = list(zip(('output', 'proj', 'z', 'core'), expected, actual))
                pairs += [(key, value, getattr(copies[1], key)) for key, value in vars(copies[0]).items() if isinstance(value, torch.Tensor)]
                for key, ref, got in pairs:
                    ref, got = ref.cpu().float(), got.cpu().float()
                    report[key] = dict(exact=torch.equal(ref, got), max_abs=(got-ref).abs().max().item())
                with Path(os.environ['QWEN_AUDIT']).open('a') as f:
                    f.write(json.dumps(report)+'\n')
            return original(self, x, past_key_value=past_key_value, **kwargs)
        qwen35.GatedDeltaNet.forward = audit
    if 'singlecommit' in parts:
        source = inspect.getsource(hpu_gated_delta._decode_deferred)
        source = source.replace('slots = ctl[2:2 + slots_max]', 'prior_rows = 1 if seq == 1 else slots_max\n    slots = ctl[2:2 + prior_rows]')
        begin = source.index('    live = pending')
        end = source.index('    committed = _commit')
        source = source[:begin] + source[begin:end].replace('slots_max', 'prior_rows') + source[end:]
        path = Path(os.environ['QWEN_AUDIT']).parent/'singlecommit_generated.py'
        path.write_text('from comfy.text_encoders.hpu_gated_delta import *\nfrom comfy.text_encoders.hpu_gated_delta import _slab, _commit, _tree_output\n'+source)
        spec = importlib.util.spec_from_file_location('qwen_singlecommit_generated', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        hpu_gated_delta._decode_deferred = module._decode_deferred
        hpu_gated_delta.compiled_ops.cache_clear()
    if 'materialized' in parts or 'tail' in parts:
        compiled = torch.compile(materialized_tail if 'tail' in parts else materialized_delta, backend='hpu_backend', dynamic=False,
                                 options={'force_static_compile': True})
        original = qwen35.GatedDeltaNet.forward
        def forward(self, x, past_key_value=None, **kwargs):
            if x.device.type == 'hpu' and x.shape[1] == 1 and type(past_key_value) is qwen35.LinearKV and past_key_value.decode_step(1):
                values = (self.in_proj_qkv(x),) if 'tail' in parts else ()
                output, *_ = compiled(self, x, past_key_value, *values)
                return output, past_key_value
            return original(self, x, past_key_value=past_key_value, **kwargs)
        qwen35.GatedDeltaNet.forward = forward
    if 'modulation' in parts:
        dit._modulated_norm = torch.compile(dit._modulated_norm, backend='hpu_backend', dynamic=False,
                                           options={'force_static_compile': True})
        dit._gated_residual = torch.compile(dit._gated_residual, backend='hpu_backend', dynamic=False,
                                           options={'force_static_compile': True})
    if 'native_sdpa' in parts:
        from habana_frameworks.torch.hpex.kernels import FusedSDPA
        def native(q, k, v, heads, mask, options, preferred):
            batch, length, width = q.shape
            q = q.view(batch, length, heads, width//heads).transpose(1, 2)
            k = k.view(batch, -1, heads, width//heads).transpose(1, 2)
            v = v.view(batch, -1, heads, width//heads).transpose(1, 2)
            result = FusedSDPA.apply(q, k, v, mask, 0.0, False, None, 'default', False)
            return result.transpose(1, 2).flatten(2)
        compiled_attention = torch.compile(native, backend='hpu_backend', dynamic=False,
                                            options={'force_static_compile': True})
        dit._compiled_attention = lambda: compiled_attention
    for part in parts:
        if part.startswith('heads'):
            group = int(part[5:])
            original_attention = dit._attention
            def grouped(q, k, v, heads, mask, options, preferred):
                outputs = []
                width = q.shape[-1] // heads
                for start in range(0, heads, group):
                    stop = min(start + group, heads)
                    output = original_attention(q[..., start*width:stop*width].contiguous(), k[..., start*width:stop*width].contiguous(),
                                                v[..., start*width:stop*width].contiguous(), stop-start, mask, options, preferred)
                    outputs.append(output.reshape(q.shape[0], q.shape[1], stop-start, -1))
                return torch.cat(outputs, dim=2).flatten(2)
            dit._compiled_attention = lambda: torch.compile(grouped, backend='hpu_backend', dynamic=False,
                                                            options={'force_static_compile': True})
            compiled_attention = dit._compiled_attention()
            dit._compiled_attention = lambda: compiled_attention
