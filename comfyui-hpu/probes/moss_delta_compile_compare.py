"""Check whether prototype/product compile options explain the timing spread."""
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import torch
from comfy.text_encoders import hpu_gated_delta

root=Path(os.environ['MOSS_NEXT_ROOT'])
spec=importlib.util.spec_from_file_location('moss_delta_compare_generated',root/'tree-delta/tree_decode_generated.py')
prototype=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=prototype
spec.loader.exec_module(prototype)
torch.set_grad_enabled(False)
torch.set_num_threads(16)
cases=[('prototype',prototype.tree_decode,dict(force_static_compile=True)),
       ('product',hpu_gated_delta._decode_deferred,{}),
       ('product-static',hpu_gated_delta._decode_deferred,dict(force_static_compile=True))]
records=[]
for capture in ('delta-0-0','delta-64-12'):
    data=torch.load(root/'baseline-capture/delta-captures'/f'{capture}.pt',weights_only=True)
    for name,function,options in cases:
        values=[v.to('hpu').clone() if isinstance(v,torch.Tensor) else v for v in data['args']]
        compiled=torch.compile(function,backend='hpu_backend',fullgraph=True,dynamic=False,options=options)
        output=compiled(*values).cpu().float()
        delta=output-data['output'].float()
        for _ in range(10):
            compiled(*values)
        torch.hpu.synchronize()
        samples=[]
        for _ in range(5):
            start=time.perf_counter()
            for _ in range(50):
                compiled(*values)
            torch.hpu.synchronize()
            samples.append((time.perf_counter()-start)*1000/50)
        records.append(dict(capture=capture,name=name,median_ms=statistics.median(samples),samples_ms=samples,
                            rrmse=float(delta.square().mean().sqrt()/data['output'].float().square().mean().sqrt())))
        (root/'delta-compile-compare.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(records[-1]),flush=True)
