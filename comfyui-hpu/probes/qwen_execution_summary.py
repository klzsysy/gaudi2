"""Summarize complete fixed-token runs and compare logits on CPU."""
import argparse
import json
from pathlib import Path
import statistics

import torch


parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--reference", type=Path)
args = parser.parse_args()
torch.set_num_threads(1)
reference_path = args.reference or args.root / "reference/logits.pt"
reference = torch.load(reference_path, map_location="cpu", weights_only=True) if reference_path.exists() else None
report = {}
for path in sorted(args.root.glob("*/timings.jsonl")):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    timed = [row for row in rows if row["mode"] == "plain"]
    result = dict(runs=rows)
    if timed:
        result.update(median_seconds=statistics.median(row["seconds"] for row in timed),
                      tokens=[row["tokens"] for row in timed],
                      median_tokens_per_second=statistics.median(row["tokens"] / row["seconds"] for row in timed))
    logits_path = path.parent / "logits.pt"
    if reference is not None and logits_path.exists():
        actual = torch.load(logits_path, map_location="cpu", weights_only=True)
        assert actual.shape == reference.shape
        squared_error = squared_reference = kl_sum = 0.0
        maximum = 0.0
        exact_rows = top1 = 0
        finite = True
        for start in range(0, len(actual), 16):
            expected = reference[start:start + 16].float()
            observed = actual[start:start + 16].float()
            difference = observed - expected
            squared_error += difference.double().square().sum().item()
            squared_reference += expected.double().square().sum().item()
            maximum = max(maximum, difference.abs().max().item())
            exact_rows += (difference == 0).all(-1).sum().item()
            top1 += (observed.argmax(-1) == expected.argmax(-1)).sum().item()
            finite = finite and observed.isfinite().all().item()
            log_expected = expected.log_softmax(-1)
            log_observed = observed.log_softmax(-1)
            kl_sum += (log_expected.exp() * (log_expected - log_observed)).sum(-1).sum().item()
        result["logits"] = dict(shape=list(actual.shape), finite=finite, rrmse=(squared_error / squared_reference) ** .5,
                                max_abs=maximum, exact_rows=exact_rows, top1_same=top1,
                                mean_kl_reference_to_candidate=kl_sum / len(actual))
    report[path.parent.name] = result
    print(path.parent.name, {key: value for key, value in result.items() if key != "runs"}, flush=True)
(args.root / "summary.json").write_text(json.dumps(report, indent=2))
