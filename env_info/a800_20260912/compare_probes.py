"""Compare compact probe artifacts; no full model weights are required."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

p=argparse.ArgumentParser()
p.add_argument('reference',type=Path)
p.add_argument('candidate',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
report={'reference':str(a.reference),'candidate':str(a.candidate),'ranks':[]}
for rank in range(4):
    ref=torch.load(a.reference/f'rank{rank}.pt',map_location='cpu',weights_only=True)
    new=torch.load(a.candidate/f'rank{rank}.pt',map_location='cpu',weights_only=True)
    assert ref['gradient_samples'].keys()==new['gradient_samples'].keys()
    x=torch.cat([v.float().flatten() for v in ref['gradient_samples'].values()])
    y=torch.cat([v.float().flatten() for v in new['gradient_samples'].values()])
    relative=float((x-y).norm()/x.norm().clamp_min(1e-12))
    cosine=float(F.cosine_similarity(x,y,dim=0))
    absolute=max(float((x.float()-y.float()).abs().max()) for x,y in zip(ref['log_probs'],new['log_probs']))
    row={'rank':rank,'gradient_sample_relative_l2':relative,'gradient_sample_cosine':cosine,
        'log_prob_max_absolute':absolute}
    row['passed'] = relative<.03 and cosine>.999 and absolute<.04
    report['ranks'].append(row)
report['passed']=all(row['passed'] for row in report['ranks'])
report['scope']='Full gradients tested on tiny models; real-model comparisons sample up to 256 values per parameter shard.'
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
raise SystemExit(0 if report['passed'] else 1)
