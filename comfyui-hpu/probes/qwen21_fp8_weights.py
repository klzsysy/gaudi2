"""Audit FP8 conversion error, preserved tensors and model file identities."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

parser = argparse.ArgumentParser()
parser.add_argument('source', type=Path)
parser.add_argument('destination', type=Path)
parser.add_argument('output', type=Path)
a = parser.parse_args()
torch.set_num_threads(16)
rows = []
with safe_open(str(a.source), framework='pt') as source, safe_open(str(a.destination), framework='pt') as destination:
    metadata = destination.metadata()
    layers = json.loads(metadata['_quantization_metadata'])['layers']
    for key in source.keys():
        original = source.get_tensor(key)
        actual = destination.get_tensor(key)
        layer = key.removesuffix('.weight')
        if layer in layers:
            scale = destination.get_tensor(layer + '.weight_scale')
            original = original.float()
            restored = actual.float() * scale
            assert restored.isfinite().all()
            rows.append(dict(key=key, shape=list(original.shape), elements=original.numel(),
                             relative_rms=float((restored-original).norm()/original.norm().clamp_min(1e-20)),
                             max_encoded=float(actual.float().abs().max())))
        else:
            assert torch.equal(original, actual), key
    result = dict(metadata=metadata, converted=rows, preserved_tensors=len(source.keys())-len(rows))
result['files'] = {}
for path in (a.source, a.destination):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16*1024*1024), b''):
            digest.update(chunk)
    result['files'][str(path)] = dict(bytes=path.stat().st_size, sha256=digest.hexdigest())
a.output.write_text(json.dumps(result, indent=2))
print(json.dumps(dict(converted=len(rows), max_relative_rms=max(r['relative_rms'] for r in rows), files=result['files'])))
