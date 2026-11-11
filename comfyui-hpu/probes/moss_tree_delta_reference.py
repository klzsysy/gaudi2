"""CPU algebra checks for arbitrary valid program permutations and zero decay."""
import json
from pathlib import Path
import sys
import torch
from moss_tree_delta import parallel_commit,parallel_tree

torch.set_num_threads(1)
torch.manual_seed(935)
records = []
for dtype in (torch.float64,torch.float32):
    for length in (1,2,3,4,8):
        for index in range(8):
            state = torch.randn(2,3,16,12,dtype=dtype)
            q = torch.nn.functional.normalize(torch.randn(2,3,16,length,dtype=dtype),dim=2)*.25
            k = torch.nn.functional.normalize(torch.randn(2,3,16,length,dtype=dtype),dim=2)
            v = torch.randn(2,3,12,length,dtype=dtype)
            decay = torch.rand(2,length,3,dtype=dtype)
            beta = torch.rand(2,length,3,dtype=dtype)
            if index%2==0:
                decay[:,0]=0
                beta[:,-1]=0
            def step(st,row):
                decayed = st*decay[:,row,:,None,None]
                delta = (v[...,row]-torch.einsum('bhk,bhkv->bhv',k[...,row],decayed))*beta[:,row,:,None]
                updated = decayed+k[...,row,None]*delta[:,:,None,:]
                return updated,torch.einsum('bhk,bhkv->bhv',q[...,row],updated)
            expected = state.clone()
            for row in range(length):
                expected,_ = step(expected,row)
            actual = parallel_commit(state,k,v,decay,beta)
            tolerance = 1e-12 if dtype==torch.float64 else 2e-6
            torch.testing.assert_close(actual,expected,atol=tolerance,rtol=tolerance)
            program = torch.randperm(length) | (torch.randint(0,2,(length,))*32)
            expected = torch.empty(2,length,3,12,dtype=dtype)
            current = state
            for instruction in program.tolist():
                row = instruction & 15
                updated,output = step(current,row)
                expected[:,row]=output
                if instruction & 32:
                    current = updated
            actual = parallel_tree(state,q,k,v,decay,beta,program)
            torch.testing.assert_close(actual,expected,atol=tolerance,rtol=tolerance)
            records.append(dict(dtype=str(dtype),length=length,index=index,max_error=float((actual-expected).abs().max())))
Path(sys.argv[1]).write_text(json.dumps(records,indent=2))
print(len(records),'algebra cases passed',max(x['max_error'] for x in records))
