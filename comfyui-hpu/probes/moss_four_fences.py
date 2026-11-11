"""Fence the asynchronous regions and save the first real four-row inputs."""
import os
import json
from pathlib import Path
import torch
from comfy.text_encoders import hpu_gated_delta,hpu_mlp


def install():
    root=Path(os.environ['MOSS_RESULT_DIR'])
    seen=set()
    conv,delta=hpu_gated_delta.compiled_ops()
    if os.environ.get('MOSS_FOUR_RAW_INPUT')=='1':
        delta=torch.compile(hpu_gated_delta._decode_deferred,backend='hpu_backend',fullgraph=True,dynamic=False)
    original_mlp=hpu_mlp.forward
    def cpu(value):
        return value.detach().cpu().clone() if isinstance(value,torch.Tensor) else value
    def fenced(name,function,values):
        original_values=values
        copied=[]
        selected=values[0].shape[1]==4
        if selected:
            print(name,'before',flush=True)
            torch.hpu.synchronize()
            if name not in seen:
                torch.save([cpu(v) for v in values],root/(name+'-four.pt'))
                (root/(name+'-metadata.json')).write_text(json.dumps([
                    dict(shape=list(v.shape),stride=list(v.stride()),offset=v.storage_offset(),requires_grad=v.requires_grad,
                         inference=v.is_inference(),type=type(v).__name__) if isinstance(v,torch.Tensor) else v for v in values],indent=2))
                seen.add(name)
            if name=='delta' and os.environ.get('MOSS_FOUR_COPY'):
                values=list(values)
                for index in map(int,os.environ['MOSS_FOUR_COPY'].split(',')):
                    values[index]=values[index].clone()
            if name=='delta' and os.environ.get('MOSS_FOUR_CANONICALIZE'):
                setting=os.environ['MOSS_FOUR_CANONICALIZE']
                copied=range(len(values)) if setting=='all' else list(map(int,setting.split(',')))
                values=[v.detach().cpu().to(v.device) if index in copied and isinstance(v,torch.Tensor) else v for index,v in enumerate(values)]
        output=function(*values)
        if selected:
            torch.hpu.synchronize()
            if name=='delta' and copied:
                for index in (5,13,14):
                    if index in copied:
                        original_values[index].copy_(values[index])
            print(name,'after',flush=True)
            if name=='delta' and os.environ.get('MOSS_FOUR_STOP_AFTER_FIRST')=='1':
                (root/'first-delta-success.json').write_text(json.dumps(dict(shape=list(values[0].shape),copied=list(copied))))
                raise SystemExit(0)
        return output
    def conv_call(*values):
        return fenced('conv',conv,values)
    def delta_call(*values):
        return fenced('delta',delta,values)
    def mlp(module,x,norm):
        selected=x.shape[1]==4
        if selected:
            print('mlp before',flush=True)
            torch.hpu.synchronize()
            if 'mlp' not in seen:
                torch.save(dict(x=cpu(x),weights={k:cpu(v) for k,v in module.state_dict().items()},
                                norm={k:cpu(v) for k,v in norm.state_dict().items()},eps=norm.eps,add=norm.add,
                                merged=module.merged_mlp),root/'mlp-four.pt')
                seen.add('mlp')
        output=original_mlp(module,x,norm)
        if selected:
            torch.hpu.synchronize()
            print('mlp after',flush=True)
        return output
    hpu_gated_delta.compiled_ops=lambda:(conv_call,delta_call)
    hpu_mlp.forward=mlp
