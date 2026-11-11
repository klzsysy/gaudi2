"""Compare pre-existing attention test failures with the committed baseline and FP64."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch
from comfy.cli_args import args
args.disable_dynamic_vram=True
args.disable_pinned_memory=True
args.disable_async_offload=True
args.use_pytorch_cross_attention=True
if os.environ.get('MOSS_AUDIT_CPU_SELECTION')=='1':
    args.cpu=True
    args.use_pytorch_cross_attention=False
args.disable_comfy_compiler=False
from comfy.ldm.moss_image import model as product
print('selected attention',product.optimized_attention.__name__,flush=True)

root=Path(os.environ['MOSS_NEXT_ROOT'])


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


baseline=load('comfy.ldm.moss_image.baseline_audit',root/'moss_baseline.py')
tests=load('moss_attention_tests',Path('tests-unit/comfy_test/hpu_moss_attention_test.py'))
torch.set_num_threads(16)
records=[]
with torch.no_grad():
    for mode,length in [('plain',31),('mask',17),('references',17)]:
        torch.manual_seed(25)
        values=tests.attention_inputs(length)
        extra={}
        if mode=='mask':
            extra['mask']=torch.randn(2,1,1,length).to(device='hpu',dtype=torch.bfloat16)*.2
        if mode=='references':
            extra['ref_kv']=torch.randn(2,2,5,128,dtype=torch.bfloat16).to('hpu')
            extra['kv_out']=torch.empty(2,2,length,128,device='hpu',dtype=torch.bfloat16)
        eager=baseline._attention(*values,**extra).cpu().double()
        results=[]
        for module in (baseline,product):
            torch._dynamo.reset()
            results.append(module.attention(*values,**extra).cpu().double())
        q,k,v,qn,kn,pe,heads,*_=values
        normed=[]
        for tensor,norm in [(q,qn),(k,kn)]:
            shaped=tensor.cpu().double().reshape(2,length,heads,-1)
            normalized=torch.nn.functional.rms_norm(shaped,(shaped.shape[-1],),norm.weight.cpu().double(),norm.eps)
            normed.append(product.rope_split_half(normalized,pe.cpu().double()).flatten(2))
        q,k=normed
        v=v.cpu().double()
        if mode=='references':
            ref=extra['ref_kv'].cpu().double()
            k,v=torch.cat([k,ref[:,0]],1),torch.cat([v,ref[:,1]],1)
        q,k,v=[x.reshape(2,-1,heads,32).transpose(1,2) for x in (q,k,v)]
        mask=extra['mask'].cpu().double() if mode=='mask' else None
        reference=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask).transpose(1,2).flatten(2)
        def error(x,y):
            delta=x-y
            return dict(rrmse=float(delta.square().mean().sqrt()/y.square().mean().sqrt()),max_abs=float(delta.abs().max()))
        record=dict(mode=mode,baseline_product_equal=torch.equal(*results),eager_fp64=error(eager,reference),
                    baseline_fp64=error(results[0],reference),product_fp64=error(results[1],reference),
                    baseline_vs_eager=error(results[0],eager))
        print(json.dumps(record),flush=True)
        records.append(record)
(root/('attention-test-audit-cpu-selection.json' if os.environ.get('MOSS_AUDIT_CPU_SELECTION')=='1' else 'attention-test-audit.json')).write_text(json.dumps(records,indent=2))
