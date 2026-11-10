"""Compare compiled Kitchen GDN with eager using captured identical inputs."""
import json
import os
from pathlib import Path
import statistics
import time
import torch
import comfy_kitchen
import habana_frameworks.torch.utils.debug as debug

torch.set_grad_enabled(False)
torch.set_num_threads(16)
directory = Path(os.environ['MOSS_CAPTURE_DIR'])
report = []
for path in sorted(directory.glob('*.pt')):
    data = torch.load(path, weights_only=True)
    eager = getattr(comfy_kitchen, data['name'])
    compiled = torch.compile(eager, backend='hpu_backend', fullgraph=True, dynamic=False)
    def inputs():
        return [v.to('hpu').clone() if isinstance(v, torch.Tensor) else v for v in data['args']]
    reference = inputs()
    candidate = inputs()
    ref_out = eager(*reference, **data['kwargs'])
    got_out = compiled(*candidate, **data['kwargs'])
    torch.hpu.synchronize()
    comparisons = []
    for name, ref, got in [('output', ref_out, got_out)] + [(str(i), a,b) for i,(a,b) in enumerate(zip(reference,candidate)) if isinstance(a,torch.Tensor)]:
        if ref is None:
            continue
        ref, got = ref.cpu(), got.cpu()
        finite = torch.isfinite(ref) & torch.isfinite(got)
        delta = (ref.float() - got.float()).abs()[finite]
        comparisons.append({'tensor': name, 'shape': list(ref.shape), 'dtype': str(ref.dtype),
                            'equal': torch.allclose(ref,got,rtol=0,atol=0,equal_nan=True),
                            'max_abs': delta.max().item() if delta.numel() else None,
                            'mean_abs': delta.mean().item() if delta.numel() else None,
                            'ref_abs_max': ref[finite].float().abs().max().item() if delta.numel() else None})
    times = {}
    for label, function in [('eager', eager), ('compile', compiled)]:
        values = inputs()
        function(*values, **data['kwargs'])
        torch.hpu.synchronize()
        samples = []
        for _ in range(10):
            start=time.perf_counter()
            function(*values, **data['kwargs'])
            torch.hpu.synchronize()
            samples.append(time.perf_counter()-start)
        times[label] = statistics.median(samples)
    record = {'file':path.name, 'compare':comparisons, 'seconds':times, 'fallback':debug._get_fallback_op_count()}
    report.append(record)
    print(json.dumps(record),flush=True)
(directory / 'replay.json').write_text(json.dumps(report,indent=2))
