"""Compare repeated complete MTP logits before/after the wider GDN region."""
import json
from pathlib import Path
import sys
import torch

root=Path(sys.argv[1])
torch.set_num_threads(1)
reference=torch.load(root/'region-logits-baseline/mtp-logits-2.pt',weights_only=True)
records={}
for name,path in [('baseline-repeat','region-logits-baseline/mtp-logits-1.pt'),
                  ('candidate','region-logits-candidate/mtp-logits-2.pt'),('candidate-cold','region-logits-candidate/mtp-logits-1.pt'),
                  ('all-vs-original','region-logits-candidate/mtp-logits-2.pt')]:
    if name=='all-vs-original':
        reference=torch.load(root/'baseline-capture/mtp-logits-1.pt',weights_only=True)
    actual=torch.load(root/path,weights_only=True)
    assert actual.shape==reference.shape and actual.isfinite().all()
    kl,top,sumsq,refsq,maximum=[],0,0.,0.,0.
    for a,b in zip(reference.split(16),actual.split(16)):
        a,b=a.float(),b.float()
        p,q=a.log_softmax(-1),b.log_softmax(-1)
        kl.extend((p.exp()*(p-q)).sum(-1).tolist())
        top+=int((a.argmax(-1)==b.argmax(-1)).sum())
        sumsq+=(a-b).square().sum().item()
        refsq+=a.square().sum().item()
        maximum=max(maximum,(a-b).abs().max().item())
    records[name]=dict(shape=list(actual.shape),top1_equal=top,rows=actual.shape[0],kl_mean=sum(kl)/len(kl),
                       kl_max=max(kl),rrmse=(sumsq/refsq)**.5,max_abs=maximum)
(root/'region-logits-comparison.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
