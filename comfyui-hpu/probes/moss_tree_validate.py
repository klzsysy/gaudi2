"""Validate real deferred state mutations and full-vocabulary accumulated logits."""
import ast
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

import torch
from comfy.text_encoders import hpu_gated_delta
import habana_frameworks.torch.utils.debug as debug

torch.set_grad_enabled(False)
torch.set_num_threads(16)
root = Path(os.environ['MOSS_NEXT_ROOT'])
kitchen = importlib.import_module('comfy_kitchen.gated_delta')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DoubleReference(ast.NodeTransformer):
    def visit_Attribute(self,node):
        self.generic_visit(node)
        if node.attr == 'float':
            node.attr = 'double'
        elif node.attr == 'float32':
            node.attr = 'float64'
        return node


source = ast.fix_missing_locations(DoubleReference().visit(ast.parse(inspect.getsource(kitchen._decode_deferred_torch))))
reference_scope = dict(torch=torch,_slab=kitchen._slab,SLOT_MAX=kitchen.SLOT_MAX)
exec(compile(source,'<fp64 decode reference>','exec'),reference_scope)
reference = reference_scope['_decode_deferred_torch']
tree = load_module('moss_tree_validate_generated',root/'tree-delta/tree_decode_generated.py')
candidate = torch.compile(tree.tree_decode,backend='hpu_backend',dynamic=False,options={'force_static_compile':True})
prefix = ''
if os.environ.get('MOSS_VALIDATE_PRODUCT') == '1':
    candidate = hpu_gated_delta.compiled_ops()[1]
    prefix = 'product-'


def clone(values, device='cpu', double=False):
    return [v.to(device=device,dtype=torch.float64 if double and v.is_floating_point() else v.dtype).clone()
            if isinstance(v,torch.Tensor) else v for v in values]


def error(actual,expected):
    actual,expected = actual.double().cpu(),expected.double().cpu()
    difference = actual-expected
    return dict(rrmse=float(difference.square().mean().sqrt()/expected.square().mean().sqrt().clamp(min=1e-30)),
                max_abs=float(difference.abs().max()),finite=bool(actual.isfinite().all()))


records = []
for path in sorted((root/'baseline-capture/delta-captures').glob('delta-*.pt')):
    captured = torch.load(path,weights_only=True)
    values = clone(captured['args'],'hpu')
    output = candidate(*values).cpu()
    state = values[5].cpu()
    double_values = clone(captured['args'],double=True)
    expected = reference(*double_values)
    record = dict(case=path.stem,shape=list(values[0].shape),
                  output_vs_baseline=error(output,captured['output']),state_vs_baseline=error(state,captured['state']),
                  output_vs_fp64=error(output,expected),baseline_output_vs_fp64=error(captured['output'],expected),
                  state_vs_fp64=error(state,double_values[5]),baseline_state_vs_fp64=error(captured['state'],double_values[5]),
                  gates_equal=torch.equal(values[13].cpu(),captured['gates']),
                  sumsq_equal=torch.equal(values[14].cpu(),captured['sumsq']))
    records.append(record)
    print(json.dumps(record),flush=True)
    (root/(prefix+'tree-numerical.json')).write_text(json.dumps(dict(cases=records,fallback=debug._get_fallback_op_count()),indent=2))

baseline = torch.load(root/'baseline-capture/mtp-logits-1.pt',weights_only=True).float()
candidate_logits = torch.load(root/'tree-logits/mtp-logits-1.pt',weights_only=True).float()
logits = error(candidate_logits,baseline)
logits.update(shape=list(baseline.shape),top1_agreement=float((candidate_logits.argmax(-1)==baseline.argmax(-1)).float().mean()))
logits['mean_kl'] = float(torch.nn.functional.kl_div(candidate_logits.log_softmax(-1),baseline.softmax(-1),reduction='batchmean'))
(root/'tree-logits-comparison.json').write_text(json.dumps(logits,indent=2))
print(json.dumps(logits),flush=True)
