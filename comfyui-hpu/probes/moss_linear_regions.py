"""Compile remaining eager dense projections without changing state math."""
import functools

import torch
import comfy.ops
from comfy.text_encoders import qwen35


def install(mode):
    original = comfy.ops.disable_weight_init.Linear.forward
    compiled = torch.compile(original,backend='hpu_backend',dynamic=None,options={'force_static_compile':True})
    if mode != 'all':
        cls = qwen35.GatedDeltaNet if mode == 'delta' else qwen35.Qwen35
        initialize = cls.__init__
        @functools.wraps(initialize)
        def init(module,*values,**kwargs):
            initialize(module,*values,**kwargs)
            children = [module.in_proj_qkv,module.in_proj_z,module.out_proj] if mode == 'delta' else [module.model.lm_head,module.mtp.fc]
            for child in children:
                child._moss_linear_region = True
        cls.__init__ = init

    @functools.wraps(original)
    def forward(module,x,*values,**kwargs):
        if torch.compiler.is_compiling():
            return original(module,x,*values,**kwargs)
        eligible = (x.device.type=='hpu' and x.dtype==torch.bfloat16 and x.ndim==3 and x.shape[1]<=8
                    and (mode=='all' or getattr(module,'_moss_linear_region',False)))
        return (compiled if eligible else original)(module,x,*values,**kwargs)
    comfy.ops.disable_weight_init.Linear.forward = forward
