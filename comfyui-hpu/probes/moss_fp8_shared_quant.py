"""Reuse one short-step activation quantization across existing CombinedLinear members."""
import json
import os
from pathlib import Path
import runpy
import torch
import comfy.ops as ops
import comfy.model_management


def install():
    original = ops.CombinedLinear.forward
    audit = os.environ.get('MOSS_SHARED_QUANT_AUDIT') == '1'
    audited = set()
    def forward(self, x):
        if audit and not torch.compiler.is_compiling() and tuple(x.shape) not in audited:
            audited.add(tuple(x.shape))
            record = dict(shape=list(x.shape), dtype=str(x.dtype), combined=self._v_weight is not None,
                          grad=torch.is_grad_enabled(), training=comfy.model_management.in_training,
                          members=[dict(layout=getattr(linear, 'layout_type', None), full_precision=linear._full_precision_mm,
                                        bias=linear.bias is not None, weight_functions=len(linear.weight_function),
                                        bias_functions=len(linear.bias_function), forced=getattr(linear, 'comfy_force_forward', False),
                                        hooks=ops.has_forward_hooks(linear), input_scale=getattr(linear, 'input_scale', None) is not None,
                                        pre_quant_scale=getattr(linear, 'pre_quant_scale', None) is not None) for linear in self.linears])
            with (Path(os.environ['MOSS_RESULT_DIR']) / 'shared-quant-calls.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
        if (self._v_weight is not None or x.device.type != 'hpu' or x.dtype != torch.bfloat16
                or x.ndim != 3 or x.shape[0] != 1 or x.shape[1] > 8
                or torch.is_grad_enabled() or comfy.model_management.in_training):
            return original(self, x)
        for linear in self.linears:
            if (getattr(linear, 'layout_type', None) != 'HPUFP8E4M3Layout' or linear._full_precision_mm
                    or linear.bias is not None or linear.weight_function or linear.bias_function
                    or getattr(linear, 'comfy_force_forward', False) or ops.has_forward_hooks(linear)
                    or getattr(linear, 'input_scale', None) is not None
                    or getattr(linear, 'pre_quant_scale', None) is not None):
                return original(self, x)
        if audit and not torch.compiler.is_compiling() and x.shape[1] not in audited:
            audited.add(x.shape[1])
            record = dict(rows=x.shape[1], members=len(self.linears), widths=[linear.weight.shape[0] for linear in self.linears])
            with (Path(os.environ['MOSS_RESULT_DIR']) / 'shared-quant-coverage.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
        ops.run_every_op()
        flat = x.reshape(-1, x.shape[-1])
        scale = flat.abs().amax(-1, keepdim=True).float().clamp_min(1e-12) / 240.0
        quantized = torch.ops.hpu.cast_to_fp8_v2(flat, scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
        outputs = []
        for linear in self.linears:
            with ops.CastBiasWeightContext(linear, x, dtype=linear.weight.dtype, bias_dtype=x.dtype,
                                          offloadable=True, compute_dtype=x.dtype, want_requant=True) as (weight, bias):
                output = torch.ops.hpu.fp8_gemm_v2(quantized, False, weight._qdata, True, None, x.dtype,
                                                 scale, weight._params.scale, bias, False)
                outputs.append(output.reshape(*x.shape[:-1], weight.shape[0]))
        return torch.cat(outputs, dim=-1)
    ops.CombinedLinear.forward = forward


if __name__ == '__main__':
    import moss_fp8_shared_quant
    moss_fp8_shared_quant.install()
    runpy.run_path(str(Path(__file__).with_name('moss_gdn_experiment.py')), run_name='__main__')
