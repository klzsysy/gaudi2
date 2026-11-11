"""Investigate omitting neutral replay slots in the installed Kitchen reference.

The generated function changes only the previous-state loop bound. All buffer
layouts, gate calculations, current tree updates and normalization stay intact.
"""
import ast
import functools
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

import torch
from comfy.text_encoders import qwen35,hpu_gated_delta


def install(mode):
    kitchen = importlib.import_module('comfy_kitchen.gated_delta')
    source = inspect.getsource(kitchen._decode_deferred_torch)
    tree = ast.parse(source)
    function = tree.body[0]
    function.name = 'bounded_decode'
    function.args.kwonlyargs.append(ast.arg(arg='commit_limit'))
    function.args.kw_defaults.append(None)
    matches = 0
    for node in ast.walk(function):
        if (isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='i'
                and isinstance(node.iter,ast.Call) and isinstance(node.iter.func,ast.Name)
                and node.iter.func.id=='range' and len(node.iter.args)==1
                and isinstance(node.iter.args[0],ast.Name) and node.iter.args[0].id=='SLOT_MAX'):
            node.iter.args[0] = ast.Name(id='commit_limit',ctx=ast.Load())
            matches += 1
    assert matches == 1
    root = Path(os.environ['MOSS_RESULT_DIR'])
    root.mkdir(parents=True,exist_ok=True)
    path = root/'bounded_decode_generated.py'
    path.write_text('import torch\nfrom comfy_kitchen.gated_delta import _slab\nSLOT_MAX = '+str(kitchen.SLOT_MAX)+'\n'+ast.unparse(ast.fix_missing_locations(tree))+'\n')
    (root/'specialization.json').write_text(json.dumps(dict(source=inspect.getfile(kitchen),source_sha256=hashlib.sha256(source.encode()).hexdigest(),mode=mode,change='previous replay loop bound only'),indent=2))
    spec = importlib.util.spec_from_file_location('moss_bounded_generated',path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    compiled = torch.compile(module.bounded_decode,backend='hpu_backend',dynamic=False,options={'force_static_compile':True})
    original_forward = qwen35.GatedDeltaNet.forward
    original_ops = hpu_gated_delta.compiled_ops()
    current_limit = 8
    current_step = None

    def delta(*values):
        return compiled(*values,commit_limit=current_limit)

    @functools.wraps(original_forward)
    def forward(layer,x,past_key_value=None,**kwargs):
        nonlocal current_limit,current_step
        if past_key_value is not None and past_key_value.decode_step(x.shape[1]):
            tracker = past_key_value.ctl_tracker
            step = (id(tracker),tracker['step'])
            if step != current_step:
                current_step = step
                pending = int(tracker['host'][tracker['parity']][0])
                current_limit = pending if mode=='exact' else 1 if pending<2 else 1<<(pending-1).bit_length()
        return original_forward(layer,x,past_key_value=past_key_value,**kwargs)
    qwen35.GatedDeltaNet.forward = forward
    hpu_gated_delta.compiled_ops = lambda:(original_ops[0],delta)
