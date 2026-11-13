"""Compare real block inputs against the eager implementation, outside timing runs."""
import json
import os
from pathlib import Path
import torch
import comfy.model_management
from comfy.ldm.qwen_image21 import model


def install():
    forward = model.QwenImage21TransformerBlock.forward
    counts = {}
    current = []
    ffn_forward = model.SwiGLUFeedForward.forward
    destination = Path(os.environ['QWEN21_COMPARE'])
    def compare(self, x, mod, pe, attn_fn, prefix_len, transformer_options={}):
        index = transformer_options.get('block_index', -1)
        if index not in (0, 16, 31) or x.shape[1] < 1024:
            return forward(self,x,mod,pe,attn_fn,prefix_len,transformer_options)
        step = counts.get(index,0)
        counts[index] = step + 1
        if step not in (0,4,7):
            return forward(self,x,mod,pe,attn_fn,prefix_len,transformer_options)
        disabled = comfy.model_management.args.disable_comfy_compiler
        try:
            comfy.model_management.args.disable_comfy_compiler = True
            expected = forward(self,x.clone(),mod,pe,attn_fn,prefix_len,transformer_options).cpu().double()
        finally:
            comfy.model_management.args.disable_comfy_compiler = disabled
        current[:] = [index, step]
        result = forward(self,x,mod,pe,attn_fn,prefix_len,transformer_options)
        current.clear()
        actual = result.cpu().double()
        error = actual - expected
        row = dict(block=index,step=step,shape=list(x.shape),finite=bool(torch.isfinite(actual).all()),
                   rrmse=float(error.square().mean().sqrt()/expected.square().mean().sqrt()),
                   maximum=float(error.abs().max()))
        with destination.open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        assert row['finite'] and row['rrmse'] < .01, row
        return result
    def compare_ffn(self, x):
        result = ffn_forward(self, x)
        if not current:
            return result
        expected = self._forward(x).cpu().double()
        actual = result.cpu().double()
        rows = torch.linspace(0, x.shape[1] - 1, 8).long()
        sample = x.cpu().double()[:, rows]
        gate, up = torch.nn.functional.linear(sample, self.gate_up.weight.cpu().double()).chunk(2, -1)
        reference = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, self.out.weight.cpu().double())
        denominator = reference.square().mean().sqrt()
        eager_error = float((expected[:, rows] - reference).square().mean().sqrt() / denominator)
        compiled_error = float((actual[:, rows] - reference).square().mean().sqrt() / denominator)
        record = dict(kind='ffn_fp64', block=current[0], step=current[1], rows=rows.tolist(),
                      eager_rrmse=eager_error, compiled_rrmse=compiled_error, ratio=compiled_error/eager_error)
        with destination.open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        return result
    model.SwiGLUFeedForward.forward = compare_ffn
    model.QwenImage21TransformerBlock.forward = compare
