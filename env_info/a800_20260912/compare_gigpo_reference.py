"""CPU arithmetic comparisons against an inspected, pinned official source file.

Only explicitly named pure function definitions are loaded through AST; imports
and other top-level statements from the downloaded file are not executed.
"""
import argparse
import ast
from collections import defaultdict,Counter
import contextlib
import hashlib
import io
import json
from pathlib import Path
import uuid
import numpy as np
import torch
from tau3_grpo.algorithms.tau_gigpo import StepRecord,compute_tau_gigpo_advantage
from tau3_grpo.algorithms.verl_estimator import compute_tau_gigpo_verl


def compare(source):
    names={'to_hashable','summarize_group_size','compute_gigpo_outcome_advantage','episode_norm_reward','build_step_group','step_norm_reward'}
    parsed=ast.parse(source.read_text());functions=[n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions}==names
    scope={'np':np,'torch':torch,'defaultdict':defaultdict,'Counter':Counter,'uuid':uuid}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),scope)
    rows=[]
    for case,lengths,rewards in [('equal_lengths',[2,2],[1.,0.]),('unequal_lengths',[2,4],[1.,0.]),('single_trajectory_repeat',[2],[1.])]:
        steps=[StepRecord(i,j,'repeat' if case=='single_trajectory_repeat' else f'anchor-{j}',(j,j+1)) for i,n in enumerate(lengths) for j in range(n)]
        flat_rewards=torch.tensor([[rewards[s.trajectory_index]] for s in steps],dtype=torch.float64)
        step_rewards=torch.tensor([.95**(lengths[s.trajectory_index]-1-s.step_index)*rewards[s.trajectory_index] for s in steps],dtype=torch.float64)
        ids=np.array(['group']*len(steps));anchors=np.array([s.anchor_id for s in steps]);mask=np.array([[int(j<n) for j in range(max(lengths))] for n in lengths])
        current,_=compute_tau_gigpo_advantage(rewards,['group']*len(lengths),steps,response_length=max(lengths),response_mask=mask)
        with contextlib.redirect_stdout(io.StringIO()):reference,_=scope['compute_gigpo_outcome_advantage'](flat_rewards,step_rewards,torch.ones_like(flat_rewards),anchors,ids,mode='mean_norm')
        actual=np.array([current[s.trajectory_index,s.step_index] for s in steps])
        difference=float(np.max(np.abs(actual-reference[:,0].numpy())))
        rows.append({'case':case,'current':actual.tolist(),'reference':reference[:,0].tolist(),'max_abs':difference})
    assert rows[0]['max_abs']<1e-7
    assert rows[1]['max_abs']>0.1
    assert rows[2]['max_abs']<1e-7
    return {'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':rows,'scope':'controlled step/trajectory layout; not full trainer equivalence','cuda_initialized':torch.cuda.is_initialized()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=compare(a.source);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
