"""Replace only Kitchen's recurrence region in an isolated experiment."""
import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import torch
from comfy.text_encoders import hpu_gated_delta


def install():
    kitchen = importlib.import_module('comfy_kitchen.gated_delta')
    source = inspect.getsource(kitchen._decode_deferred_torch)
    tree = ast.parse(source)
    function = tree.body[0]
    function.name = 'tree_decode'
    start = next(i for i,node in enumerate(function.body) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='st' for t in node.targets))
    end = next(i for i,node in enumerate(function.body) if i>start and isinstance(node,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='oc' for t in node.targets)
               and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='float')
    replacement = ast.parse('''
q,k = normalized(prev_q,prev_k,prev_ss)
st = parallel_commit(state,k,prev_v,prev_decay,prev_beta)
state.copy_(st)
q,k = normalized(cur_q,cur_k,cur_ss)
oc = parallel_tree(st,q,k,cur_v,decay,beta,prog[:seq])
''').body
    function.body[start:end] = replacement
    root = Path(os.environ['MOSS_RESULT_DIR'])
    root.mkdir(parents=True,exist_ok=True)
    path = root/'tree_decode_generated.py'
    path.write_text('import torch\nfrom comfy_kitchen.gated_delta import _slab\nfrom moss_tree_delta import parallel_commit,parallel_tree\nSLOT_MAX = '+str(kitchen.SLOT_MAX)+'\n'+ast.unparse(ast.fix_missing_locations(tree))+'\n')
    (root/'specialization.json').write_text(json.dumps(dict(source=inspect.getfile(kitchen),source_sha256=hashlib.sha256(source.encode()).hexdigest(),change='parallel recurrence, original gate/buffer/normalization preparation'),indent=2))
    spec = importlib.util.spec_from_file_location('moss_tree_generated',path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original = hpu_gated_delta.compiled_ops()
    compiled = torch.compile(module.tree_decode,backend='hpu_backend',dynamic=False,options={'force_static_compile':True})
    hpu_gated_delta.compiled_ops = lambda:(original[0],compiled)
