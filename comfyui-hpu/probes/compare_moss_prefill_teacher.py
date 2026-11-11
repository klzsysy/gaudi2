"""CPU-only logits and timing comparison for the same teacher trajectory."""
import json
from pathlib import Path
import statistics
import sys

import torch

torch.set_num_threads(8)
root = Path(sys.argv[1])
cases = {}
all_ids = []
for name in ('baseline', 'bucketed'):
    directory = root / ('teacher-' + name)
    runs = [json.loads(s) for s in (directory / 'runs.jsonl').read_text().splitlines()]
    warm = [r for r in runs if r['warm']]
    assert len(warm) == 5
    assert all(len(r['ids']) == 298 and not r['fallback'] for r in runs)
    all_ids.extend(r['ids'] for r in runs)
    seconds = [r['seconds'] for r in warm]
    cases[name] = dict(median=statistics.median(seconds), seconds=seconds, cold=runs[0]['seconds'])
assert all(ids == all_ids[0] for ids in all_ids)
reference = torch.load(root / 'teacher-baseline/logits-1.pt', weights_only=True, map_location='cpu').float()
actual = torch.load(root / 'teacher-bucketed/logits-1.pt', weights_only=True, map_location='cpu').float()
assert reference.shape == actual.shape and reference.shape[0] == 298
assert actual.isfinite().all()
logp, logq = reference.log_softmax(-1), actual.log_softmax(-1)
kl = (logp.exp() * (logp - logq)).sum(-1)
result = dict(cases=cases, token_ids_equal=True, top1_equal=int((reference.argmax(-1) == actual.argmax(-1)).sum()),
              tokens=298, mean_kl=float(kl.mean()), maximum_kl=float(kl.max()),
              mean_logit_abs_difference=float((reference-actual).abs().mean()),
              delta_percent=100*(cases['bucketed']['median']/cases['baseline']['median']-1),
              scope='MTP off; fixed 298-token teacher; includes identical CPU logits collection in both arms')
repeat = torch.load(root / 'teacher-baseline/logits-2.pt', weights_only=True, map_location='cpu').float()
repeat_logp = repeat.log_softmax(-1)
result['baseline_repeat'] = dict(top1_equal=int((reference.argmax(-1) == repeat.argmax(-1)).sum()),
                               mean_kl=float((logp.exp() * (logp-repeat_logp)).sum(-1).mean()))
(root / 'teacher-comparison.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
