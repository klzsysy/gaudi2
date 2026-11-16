"""Audit direct INT8/FP32/FP8 PE conversion against the BF16-intermediate model."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE, dequantize_int8_convrot_weight_dtype

parser=argparse.ArgumentParser()
parser.add_argument('source',type=Path)
parser.add_argument('previous',type=Path)
parser.add_argument('current',type=Path)
parser.add_argument('output',type=Path)
a=parser.parse_args()
torch.set_num_threads(16)
rows=[]
reference_squared=old_squared=new_squared=0.
changed=elements=0
with safe_open(str(a.source),framework='pt') as source, safe_open(str(a.previous),framework='pt') as old, safe_open(str(a.current),framework='pt') as new:
    layers=json.loads(new.metadata()['_quantization_metadata'])['layers']
    assert layers==json.loads(old.metadata()['_quantization_metadata'])['layers']
    assert set(new.keys())==set(old.keys())
    preserved=0
    for key in new.keys():
        prefix=key.removesuffix('.weight')
        if prefix not in layers:
            if not key.endswith('.weight_scale'):
                actual, expected = new.get_tensor(key), old.get_tensor(key)
                assert actual.dtype == expected.dtype and torch.equal(actual, expected), key
                preserved+=1
            continue
        config=json.loads(source.get_tensor(prefix+'.comfy_quant').numpy().tobytes())
        reference=dequantize_int8_convrot_weight_dtype(source.get_tensor(key),source.get_tensor(prefix+'.weight_scale').float(),config.get('convrot_groupsize',256),DTYPE_TO_CODE[torch.float32])
        new_scale=new.get_tensor(prefix+'.weight_scale')
        expected_scale=reference.abs().amax().clamp_min(1e-12)/224.
        assert torch.equal(new_scale,expected_scale),key
        new_weight=new.get_tensor(key)
        assert torch.equal(new_weight.view(torch.uint8),(reference/new_scale).to(torch.float8_e4m3fn).view(torch.uint8)),key
        old_weight=old.get_tensor(key)
        old_value=old_weight.float()*old.get_tensor(prefix+'.weight_scale')
        new_value=new_weight.float()*new_scale
        denominator=float(reference.square().sum(dtype=torch.float64))
        old_error=float((old_value-reference).square().sum(dtype=torch.float64))
        new_error=float((new_value-reference).square().sum(dtype=torch.float64))
        different=int((old_weight.view(torch.uint8)!=new_weight.view(torch.uint8)).sum())
        rows.append(dict(key=key,elements=reference.numel(),changed_codes=different,
                         old_rrmse=(old_error/denominator)**.5,new_rrmse=(new_error/denominator)**.5))
        reference_squared+=denominator
        old_squared+=old_error
        new_squared+=new_error
        changed+=different
        elements+=reference.numel()
    result=dict(converted_layers=len(layers),preserved_tensors=preserved,old_rrmse=(old_squared/reference_squared)**.5,
                new_rrmse=(new_squared/reference_squared)**.5,changed_codes=changed,elements=elements,
                changed_fraction=changed/elements,improved_layers=sum(row['new_rrmse']<row['old_rrmse'] for row in rows),layers=rows)
result['files']={}
for path in (a.source,a.previous,a.current):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(16*1024*1024),b''):digest.update(chunk)
    result['files'][str(path)]=dict(bytes=path.stat().st_size,sha256=digest.hexdigest())
a.output.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='layers'},indent=2))
