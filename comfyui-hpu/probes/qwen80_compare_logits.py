"""Compare saved fixed-trajectory logits in bounded CPU chunks."""
import argparse
import json
import math
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument('reference', type=Path)
parser.add_argument('candidates', type=Path, nargs='+')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--tokens', type=int)
args = parser.parse_args()
torch.set_num_threads(1)
reference = torch.load(args.reference, weights_only=True, mmap=True, map_location='cpu')
ids_file = args.reference.parent / 'plain-tokens.json'
valid_tokens = args.tokens or (len(json.loads(ids_file.read_text())) if ids_file.exists() else reference.shape[0])
reference = reference[:valid_tokens]
rows = []
for path in args.candidates:
    candidate = torch.load(path, weights_only=True, mmap=True, map_location='cpu')
    candidate = candidate[:valid_tokens]
    assert candidate.shape == reference.shape, (path, candidate.shape, reference.shape)
    squared_error = squared_reference = kl = 0.
    max_abs = 0.
    matches = 0
    exact = True
    for start in range(0, reference.shape[0], 16):
        expected = reference[start:start + 16].float()
        actual = candidate[start:start + 16].float()
        difference = actual - expected
        exact = exact and torch.equal(actual, expected)
        squared_error += difference.square().sum().item()
        squared_reference += expected.square().sum().item()
        max_abs = max(max_abs, difference.abs().max().item())
        matches += (actual.argmax(-1) == expected.argmax(-1)).sum().item()
        expected_log = expected.log_softmax(-1)
        actual_log = actual.log_softmax(-1)
        kl += (expected_log.exp() * (expected_log - actual_log)).sum().item()
    row = dict(path=str(path), shape=list(reference.shape), exact=exact,
               rrmse=math.sqrt(squared_error / squared_reference), max_abs=max_abs,
               top1_matches=matches, tokens=reference.shape[0], mean_kl=kl / reference.shape[0])
    rows.append(row)
    print(row, flush=True)
    args.output.write_text(json.dumps(rows, indent=2))
