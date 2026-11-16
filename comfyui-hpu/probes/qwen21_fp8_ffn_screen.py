"""Compare compiled FP8 FFN with eager execution and CPU FP64 samples."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch
import comfy.ops
from comfy.ldm.qwen_image21 import model as dit

parser=argparse.ArgumentParser()
parser.add_argument('root',type=Path)
a=parser.parse_args()
torch.set_grad_enabled(False)
torch.set_num_threads(32)
rows=[]
for path in sorted(a.root.glob('ffn-*.pt')):
    saved=torch.load(path,weights_only=False)
    state=saved['state']
    x=saved['x'].to('hpu')
    hidden=state['gate_up.weight'].shape[0]//2
    ops=comfy.ops.mixed_precision_ops({'formats':['float8_e4m3fn']},torch.bfloat16,load_device=torch.device('hpu'))
    module=dit.SwiGLUFeedForward(x.shape[-1],hidden,device='cpu',dtype=torch.bfloat16,operations=ops)
    module.load_state_dict(state)
    module.to('hpu')
    indices=torch.linspace(0,x.shape[1]-1,32).long()
    sample=saved['x'][:,indices].double()
    w1=state['gate_up.weight'].double()*state['gate_up.weight_scale'].double()
    gate,up=torch.nn.functional.linear(sample,w1).chunk(2,dim=-1)
    del w1
    w2=state['out.weight'].double()*state['out.weight_scale'].double()
    reference=torch.nn.functional.linear(torch.nn.functional.silu(gate)*up,w2)
    del w2,gate,up
    outputs={}
    for name,function in [('eager',module._forward),('compiled',lambda x:dit._compiled_ffn()(module,x))]:
        output=function(x)
        torch.hpu.synchronize()
        outputs[name]=output.cpu().float()
        times=[]
        for _ in range(5):
            torch.hpu.synchronize();start=time.perf_counter()
            for repeat in range(4):output=function(x)
            torch.hpu.synchronize();times.append((time.perf_counter()-start)/4)
        actual=outputs[name][:,indices].double()
        rows.append(dict(file=path.name,mode=name,seconds=statistics.median(times),
            fp64_sample_rrmse=float((actual-reference).norm()/reference.norm()),finite=bool(output.isfinite().all())))
    delta=float((outputs['compiled']-outputs['eager']).norm()/outputs['eager'].norm())
    rows[-1]['relative_to_eager']=delta
    print(json.dumps(rows[-2:]),flush=True)
    del module,x,saved,state,outputs
    gc.collect()
(a.root/'ffn-screen.json').write_text(json.dumps(rows,indent=2))
