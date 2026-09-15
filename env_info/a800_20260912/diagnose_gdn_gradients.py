"""Compare every language-parameter gradient on one fixed real trajectory.

No optimizer, model update, simulator or RL training. References are retained
only in CPU RAM, not as checkpoint files. The objective is the first-update
policy-gradient term with fixed recorded advantages and policy-token masking.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoModelForImageTextToText
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from verl import DataProto
from verl.utils.qwen35_padding import install_qwen35_padding_guard
from diagnose_gdn_model import actor, kernel, compare


def main(args):
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    batch=DataProto.load_from_disk(args.batch).batch
    active=(batch['advantages']*batch['response_mask']).count_nonzero(dim=-1)
    candidates=active.nonzero().flatten().tolist()
    if not candidates:
        raise ValueError('Recorded batch has no nonzero policy advantages; cannot validate gradients.')
    if args.row is None:
        args.row=min(candidates,key=lambda i:int(batch['attention_mask'][i].sum()))
    if args.row not in candidates:
        raise ValueError(f'Row {args.row} has no nonzero policy advantages; valid rows: {candidates}')
    print(json.dumps({'selected_row':args.row,'valid_rows':candidates,
        'nonzero_advantage_tokens':int(active[args.row])}),flush=True)
    keys=['input_ids','attention_mask','position_ids','responses','response_mask','advantages']
    data={k:batch[k][args.row:args.row+1].cuda() for k in keys}
    del batch
    mask=data['response_mask']
    if args.triton_fp32!='default':
        os.environ['TRITON_F32_DEFAULT']=args.triton_fp32
        import triton.language as tl
        importlib.import_module('fla.ops.gated_delta_rule.chunk_fwd').SOLVE_TRIL_DOT_PRECISION=tl.constexpr(args.triton_fp32)
    hf.FusedRMSNormGated=None
    model=AutoModelForImageTextToText.from_pretrained(args.model,dtype=torch.float32,
        attn_implementation='sdpa',local_files_only=True).cuda()
    model.model.visual.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    install_qwen35_padding_guard(model)
    modules=[m for m in model.modules() if isinstance(m,hf.Qwen3_5GatedDeltaNet)]
    a=actor(model);a.param_dtype=torch.float32
    os.environ.update(VERL_QWEN35_COMPACT_HEAD='1',VERL_QWEN35_COMPACT_CHUNK='256',
        VERL_QWEN35_COMPACT_BACKEND='checkpoint',VERL_QWEN35_TRIM_PADDING='experimental_both')
    report={'kind':'one_real_trajectory_full_gradient_comparison_not_RL','row':args.row,'batch':args.batch,
        'full_fp32':True,'triton_fp32':args.triton_fp32,'policy_tokens':int(mask.sum()),
        'active_tokens':int(data['attention_mask'].sum()),'modes':{}}
    reference={};reference_logp=None
    for backend,trim in [('native_fp32','none'),('native_fp32','experimental_both'),('fla_fp32','experimental_both')]:
        name=backend+'/'+trim
        os.environ['VERL_QWEN35_TRIM_PADDING']=trim
        print(json.dumps({'starting':name}),flush=True)
        for module in modules:module.chunk_gated_delta_rule=kernel(backend)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        started=time.monotonic()
        out=a._forward_micro_batch(data,1.)['log_probs']
        loss=-(out*data['advantages']*mask).sum()/mask.sum()
        loss.backward();torch.cuda.synchronize()
        row={'forward_backward_seconds':time.monotonic()-started,'loss':float(loss.detach()),
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
        scored=out.detach().cpu()[mask.bool().cpu()]
        total_ref_sq=total_new_sq=total_diff_sq=total_dot=0.
        parameter_reports={};count=0
        for parameter_name,p in model.named_parameters():
            if p.grad is None:continue
            assert torch.isfinite(p.grad).all(),parameter_name
            count+=p.numel()
            values=p.grad.detach().cpu()
            if not reference_logp is not None:reference[parameter_name]=values
            else:
                ref=reference[parameter_name]
                rsq=nsq=dsq=dot=0.
                # Bounded FP64 reductions, including embedding and output head.
                for x,y in zip(ref.reshape(-1).split(1048576),values.reshape(-1).split(1048576)):
                    x,y=x.double(),y.double()
                    rsq+=float(x.square().sum());nsq+=float(y.square().sum())
                    dsq+=float((x-y).square().sum());dot+=float((x*y).sum())
                parameter_reports[parameter_name]={'relative_l2':(dsq/max(rsq,1e-30))**.5}
                total_ref_sq+=rsq;total_new_sq+=nsq;total_diff_sq+=dsq;total_dot+=dot
        row['gradient_elements_compared']=count
        if reference_logp is None:reference_logp=scored
        else:
            row['gradient_relative_l2']=(total_diff_sq/total_ref_sq)**.5
            row['gradient_cosine']=total_dot/(total_ref_sq*total_new_sq)**.5
            row['log_probs']=compare(reference_logp,scored)
            row['worst_parameters']=sorted(parameter_reports.items(),key=lambda kv:kv[1]['relative_l2'],reverse=True)[:10]
            row['passed']=row['gradient_relative_l2']<.03 and row['gradient_cosine']>.999 and row['log_probs']['max_abs']<.04
        report['modes'][name]=row
        args.output.write_text(json.dumps(report,indent=2))
        print(json.dumps({'mode':name,**row}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='/root/autodl-fs/tau3-core/code/checkpoints/sft-merged/new-off')
    p.add_argument('--batch',default='/root/autodl-fs/tau3-core/code/results/runs/rl-formal-full-20260912/acceptance/update-batches/update_000001.pkl')
    p.add_argument('--row',type=int,default=None)
    p.add_argument('--triton-fp32',choices=['default','ieee','tf32x3'],default='default')
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
