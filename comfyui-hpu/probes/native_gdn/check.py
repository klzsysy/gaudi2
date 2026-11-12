import json
import os
from pathlib import Path
import statistics
import time
import torch
from comfy.text_encoders import hpu_gated_delta as delta
from ops import core
import habana_frameworks.torch.utils.debug as debug

torch.set_grad_enabled(False)
torch.set_num_threads(32)
torch.manual_seed(43)
root=Path(os.environ['MOSS_OPT_ROOT'])


def baseline(state,pk,pv,pd,pb,q,k,v,d,b,program,pending):
    committed=delta._commit(state,pk,pv,pd,pb)
    return committed,delta._tree_output(committed,q,k,v,d,b,program)


reference_fn=torch.compile(baseline,backend='hpu_backend',dynamic=False)
candidate=torch.compile(core,backend='hpu_backend',dynamic=False)
records=[]
for batch,heads,rows,live in [(1,2,1,8),(1,2,4,8),(2,2,8,8),(1,32,8,8),(1,32,8,3),(1,32,8,0)]:
    state=torch.randn(batch,heads,128,128,device='hpu')*.01
    pk=torch.randn(batch,heads,128,8,device='hpu')*.05
    pv=torch.randn(batch,heads,128,8,device='hpu')*.1
    pd=torch.rand(batch,8,heads,device='hpu')
    pb=torch.rand(batch,8,heads,device='hpu')
    pk[...,live:]=0
    pv[...,live:]=0
    pd[:,live:]=1
    pb[:,live:]=0
    pending=torch.tensor(live,device='hpu',dtype=torch.int32)
    q=torch.randn(batch,heads,128,rows,device='hpu')*.05
    k=torch.randn_like(q)*.05
    v=torch.randn(batch,heads,128,rows,device='hpu')*.1
    d=torch.rand(batch,rows,heads,device='hpu')
    b=torch.rand(batch,rows,heads,device='hpu')
    for mode in ['chain','leaves','reordered']:
        order=list(range(rows)) if mode!='reordered' else list(reversed(range(rows)))
        program=torch.tensor([row+(32 if mode!='leaves' or index==0 else 0) for index,row in enumerate(order)],device='hpu',dtype=torch.int64)
        inputs=(state,pk,pv,pd,pb,q,k,v,d,b,program,pending)
        expected=reference_fn(*inputs)
        actual=candidate(*inputs)
        errors=[]
        for x,y in zip(actual,expected):
            a,e=x.cpu(),y.cpu()
            torch.testing.assert_close(a,e,atol=2e-6,rtol=5e-5)
            errors.append(float((a-e).norm()/e.norm()))
        durations={}
        for name,function in [('baseline',reference_fn),('candidate',candidate)]:
            for _ in range(3):function(*inputs)
            torch.hpu.synchronize()
            samples=[]
            for _ in range(3):
                start=time.perf_counter_ns()
                for _ in range(20):function(*inputs)
                torch.hpu.synchronize()
                samples.append((time.perf_counter_ns()-start)/1e9/20)
            durations[name]=statistics.median(samples)
        record=dict(batch=batch,heads=heads,rows=rows,pending=live,mode=mode,rrmse=errors,seconds=durations,fallback=debug._get_fallback_op_count())
        records.append(record)
        (root/'gdn-core-screen.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(record),flush=True)
