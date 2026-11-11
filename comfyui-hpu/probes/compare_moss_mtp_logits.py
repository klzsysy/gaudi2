"""Compare logits on identical MTP rows, with baseline repeat control."""
import json
from pathlib import Path
import sys
import torch

torch.set_num_threads(1)
root = Path(sys.argv[1])
reference = torch.load(root/'mtp-baseline-logits/mtp-logits-2.pt', map_location='cpu', weights_only=True)
results = {}
for name, file in [('baseline_repeat','mtp-baseline-logits/mtp-logits-1.pt'), ('candidate','mtp-norm-product-logits/mtp-logits-2.pt'), ('candidate_repeat','mtp-norm-product-logits/mtp-logits-1.pt')]:
    actual = torch.load(root/file, map_location='cpu', weights_only=True)
    assert actual.shape == reference.shape
    kl, top, sumsq, refsq, absmax = [], 0, 0., 0., 0.
    for a, b in zip(reference.split(16), actual.split(16)):
        a, b = a.float(), b.float()
        p, q = a.log_softmax(-1), b.log_softmax(-1)
        kl.extend((p.exp()*(p-q)).sum(-1).tolist())
        top += int((a.argmax(-1)==b.argmax(-1)).sum())
        sumsq += (a-b).square().sum().item()
        refsq += a.square().sum().item()
        absmax = max(absmax, (a-b).abs().max().item())
    results[name] = dict(shape=list(actual.shape), top1_equal=top, rows=actual.shape[0], kl_mean=sum(kl)/len(kl), kl_max=max(kl), rrmse=(sumsq/refsq)**.5, max_abs=absmax)
(root/'mtp-logits-comparison.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
