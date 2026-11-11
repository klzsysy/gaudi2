"""Compare fixed-MTP logits in bounded CPU chunks."""
import json
from pathlib import Path
import sys
import torch

torch.set_num_threads(4)
root = Path(sys.argv[1])
reference = torch.load(root / 'baseline/mtp/mtp-logits-7.pt', weights_only=True, map_location='cpu')
results = {}
for name in sys.argv[2:]:
    actual = torch.load(root / name / 'mtp/mtp-logits-7.pt', weights_only=True, map_location='cpu')
    assert actual.shape == reference.shape
    divergences, equal, squared, reference_squared, maximum = [], 0, 0.0, 0.0, 0.0
    for a, b in zip(reference.split(16), actual.split(16)):
        a, b = a.float(), b.float()
        p, q = a.log_softmax(-1), b.log_softmax(-1)
        divergences.extend((p.exp() * (p-q)).sum(-1).tolist())
        equal += (a.argmax(-1) == b.argmax(-1)).sum().item()
        squared += (a-b).square().sum().item()
        reference_squared += a.square().sum().item()
        maximum = max(maximum, (a-b).abs().max().item())
    results[name] = {'shape': list(actual.shape), 'top1_equal': equal, 'kl_mean': sum(divergences)/len(divergences),
                     'kl_max': max(divergences), 'rrmse': (squared/reference_squared)**.5, 'max_abs': maximum}
(root / 'logits-comparison.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
