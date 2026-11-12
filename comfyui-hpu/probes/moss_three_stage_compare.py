"""Compare fixed-program logits without retaining full float copies."""
import argparse
import json
from pathlib import Path

import torch


parser = argparse.ArgumentParser()
parser.add_argument('baseline', type=Path)
parser.add_argument('candidate', type=Path)
parser.add_argument('output', type=Path)
options = parser.parse_args()
torch.set_num_threads(4)
baseline = torch.load(options.baseline, weights_only=True, map_location='cpu')
candidate = torch.load(options.candidate, weights_only=True, map_location='cpu')
assert baseline.shape == candidate.shape
squared = reference_squared = maximum = 0.0
differing = top1_equal = 0
for expected, actual in zip(baseline.split(16), candidate.split(16)):
    differing += int((expected != actual).sum())
    top1_equal += int((expected.argmax(-1) == actual.argmax(-1)).sum())
    delta = actual.float() - expected.float()
    squared += float(delta.square().sum(dtype=torch.float64))
    reference_squared += float(expected.float().square().sum(dtype=torch.float64))
    maximum = max(maximum, float(delta.abs().max()))
result = dict(baseline=str(options.baseline), candidate=str(options.candidate), shape=list(baseline.shape),
              equal=differing == 0, differing=differing, top1_equal=top1_equal,
              rrmse=(squared / reference_squared) ** 0.5, maximum=maximum)
options.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result))
