"""CPU verification of the production estimator on saved historical reward groups."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tau3_grpo.algorithms.verl_estimator import compute_tau_gigpo_verl
from tau3_grpo.tracking.signal_audit import source_hashes
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def validate(snapshot):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not torch.cuda.is_initialized() and torch.cuda.device_count() == 0
    torch.set_num_threads(2)
    rows=[]
    for arm,data in snapshot['arms'].items():
        for update in data['updates']:
            group=update['rows'];size=len(group)
            rewards=torch.zeros(size,8,dtype=torch.float64)
            rewards[:,-1]=torch.tensor([r['score'] for r in group],dtype=torch.float64)
            mask=torch.tensor([[int(j<2+i%5 and j%3!=1) for j in range(8)] for i in range(size)])
            uids=np.array([r['task_id'] for r in group])
            nt={'anchor_ids':[['probe-'+str(u)] for u in uids], 'anchor_spans':[[[0,8]] for _ in uids]}
            cfg=OmegaConf.create({'gigpo':{'omega':0.,'episode_normalization':'grpo'},'norm_adv_by_std_in_grpo':True})
            actual=compute_tau_gigpo_verl(rewards,mask,index=uids,non_tensor_batch=nt,config=cfg)
            expected=compute_grpo_outcome_advantage(rewards,mask,uids,norm_adv_by_std_in_grpo=True)
            assert all(torch.equal(a,b) for a,b in zip(actual,expected,strict=True))
            rows.append({'arm':arm,'step':update['step'],'advantage_max_abs':float((actual[0]-expected[0]).abs().max()),
                         'returns_max_abs':float((actual[1]-expected[1]).abs().max())})
    return {'updates':rows,'update_count':len(rows),'cuda_initialized':torch.cuda.is_initialized(),
            'cuda_device_count':torch.cuda.device_count(),'tensor_device':'cpu','source_sha256':source_hashes(),
            'scope':'production estimator; historical rewards/groups with synthetic masks/spans, not original token batches'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=validate(json.loads(a.snapshot.read_text()));result['snapshot_sha256']=hashlib.sha256(a.snapshot.read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(result['update_count'],'updates exactly match GRPO')
