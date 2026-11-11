"""Compute shared convolution indices once per decode step on the host."""
import ast
import functools
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import torch
from comfy.text_encoders import qwen35,hpu_gated_delta


def install():
    kitchen = importlib.import_module('comfy_kitchen.gated_delta')
    tree = ast.parse(inspect.getsource(kitchen._conv_deferred_torch))
    function = tree.body[0]
    function.name = 'prepared_conv'
    function.args.kwonlyargs.append(ast.arg(arg='indices'))
    function.args.kw_defaults.append(None)
    function.body.insert(0,ast.parse('conv_w = conv_w.squeeze(1)').body[0])
    start = next(i for i,node in enumerate(function.body) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='rows' for t in node.targets))
    end = next(i for i,node in enumerate(function.body) if i>start and isinstance(node,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='src' for t in node.targets)
               and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='stack')
    function.body[start:end+1] = ast.parse('src = indices[window:window+seq*taps].reshape(seq,taps)').body
    for node in function.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='committed' for t in node.targets):
            node.value = ast.parse('combined.index_select(2,indices[:window])',mode='eval').body
    root = Path(os.environ['MOSS_RESULT_DIR'])
    root.mkdir(parents=True,exist_ok=True)
    path = root/'prepared_conv_generated.py'
    path.write_text('import torch\nfrom comfy_kitchen.gated_delta import _slab\nSLOT_MAX = '+str(kitchen.SLOT_MAX)+'\n'+ast.unparse(ast.fix_missing_locations(tree))+'\n')
    spec = importlib.util.spec_from_file_location('moss_prepared_conv_generated',path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    compiled = torch.compile(module.prepared_conv,backend='hpu_backend',dynamic=False,options={'force_static_compile':True})
    original_forward = qwen35.GatedDeltaNet.forward
    original_ops = hpu_gated_delta.compiled_ops()
    current = None

    def conv(*values):
        return compiled(*values,indices=current)

    @functools.wraps(original_forward)
    def forward(layer,x,past_key_value=None,**kwargs):
        nonlocal current
        if past_key_value is not None and past_key_value.decode_step(x.shape[1]):
            tracker = past_key_value.ctl_tracker
            window = past_key_value.conv_state.shape[2]
            taps = layer.conv_kernel_size
            step = (tracker['step'],window,taps)
            if tracker.get('prepared_conv_step') != step:
                host = tracker['host'][tracker['parity']]
                pending = int(host[0])
                parents = host[2+kitchen.SLOT_MAX:2+2*kitchen.SLOT_MAX].tolist()
                indices = list(range(pending,pending+window))
                for row in range(x.shape[1]):
                    indices.append(window+row)
                    parent,missing = row,0
                    for _ in range(1,taps):
                        parent = parents[parent] if parent>=0 else -1
                        missing += int(parent<0)
                        indices.append(window+parent if parent>=0 else window-missing)
                tracker['prepared_conv_indices'] = torch.tensor(indices,dtype=torch.long,device=x.device)
                tracker['prepared_conv_step'] = step
            current = tracker['prepared_conv_indices']
        try:
            return original_forward(layer,x,past_key_value=past_key_value,**kwargs)
        finally:
            current = None
    qwen35.GatedDeltaNet.forward = forward
    hpu_gated_delta.compiled_ops = lambda:(conv,original_ops[1])
