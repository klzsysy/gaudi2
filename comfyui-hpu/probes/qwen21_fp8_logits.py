"""Compare same-prefix FP8 PE logits in bounded CPU chunks."""
import argparse
import json
from pathlib import Path

import torch

parser=argparse.ArgumentParser()
parser.add_argument('root',type=Path)
a=parser.parse_args()
torch.set_num_threads(4)
reference=torch.load(a.root/'quality-reference/logits.pt',map_location='cpu',weights_only=True)
actual=torch.load(a.root/'quality-decode/logits.pt',map_location='cpu',weights_only=True)
assert reference.shape==actual.shape
squared=reference_squared=0.0
equal=0
maximum=0.0
divergences=[]
for left,right in zip(reference.split(16),actual.split(16)):
    left,right=left.float(),right.float()
    p,q=left.log_softmax(-1),right.log_softmax(-1)
    divergences.extend((p.exp()*(p-q)).sum(-1).tolist())
    equal+=(left.argmax(-1)==right.argmax(-1)).sum().item()
    squared+=(left-right).square().sum().item()
    reference_squared+=left.square().sum().item()
    maximum=max(maximum,(left-right).abs().max().item())
report=dict(shape=list(reference.shape),rrmse=(squared/reference_squared)**.5,top1_equal=equal,
            kl_mean=sum(divergences)/len(divergences),kl_max=max(divergences),max_abs=maximum)
(a.root/'logits-comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
