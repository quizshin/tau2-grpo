"""Warm fixed-trajectory forward/backward benchmark; no optimizer or RL.

One warmup per mode is excluded. Identical recorded tokens, advantages, policy
mask, FP32 parameter storage, native norm, and checkpointed compact head.
Full-FP32 modes change compute precision and are labeled as such.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoModelForImageTextToText
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from verl import DataProto
from verl.utils.qwen35_padding import install_qwen35_padding_guard
from diagnose_gdn_model import actor, kernel


def main(args):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    batch=DataProto.load_from_disk(args.batch).batch
    valid=(batch['advantages']*batch['response_mask']).count_nonzero(dim=-1).nonzero().flatten().tolist()
    if not valid:
        raise ValueError('No nonzero recorded policy advantages.')
    row=min(valid,key=lambda i:int(batch['attention_mask'][i].sum())) if args.row is None else args.row
    if row not in valid:
        raise ValueError('Requested row has no nonzero recorded policy advantages.')
    keys=['input_ids','attention_mask','position_ids','responses','response_mask','advantages']
    data={k:batch[k][row:row+1].cuda() for k in keys}
    del batch
    report={'kind':'warm_real_trajectory_forward_backward_NOT_full_RL_step','batch':args.batch,'row':row,
        'active_tokens':int(data['attention_mask'].sum()),'policy_tokens':int(data['response_mask'].sum()),
        'padded_tokens':data['input_ids'].shape[-1],'warmup_per_mode':1,'iterations':args.iterations,
        'gpu':torch.cuda.get_device_name(),'modes':{}}
    print(json.dumps(report),flush=True)
    hf.FusedRMSNormGated=None
    model=AutoModelForImageTextToText.from_pretrained(args.model,dtype=torch.float32,
        attn_implementation='sdpa',local_files_only=True).cuda()
    model.model.visual.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    install_qwen35_padding_guard(model)
    modules=[m for m in model.modules() if isinstance(m,hf.Qwen3_5GatedDeltaNet)]
    a=actor(model)
    os.environ.update(VERL_QWEN35_COMPACT_HEAD='1',VERL_QWEN35_COMPACT_CHUNK='256',
        VERL_QWEN35_COMPACT_BACKEND='checkpoint')
    for backend,trim,precision in [('native_autocast','none','bf16'),
        ('native_fp32','experimental_both','fp32'),('fla_fp32','experimental_both','fp32')]:
        name=f'{backend}/{trim}/{precision}'
        a.param_dtype=torch.bfloat16 if precision=='bf16' else torch.float32
        os.environ['VERL_QWEN35_TRIM_PADDING']=trim
        for module in modules:module.chunk_gated_delta_rule=kernel(backend)
        times=[];memory=[];losses=[]
        print(json.dumps({'starting':name}),flush=True)
        for iteration in range(args.iterations+1):
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            start=time.monotonic()
            out=a._forward_micro_batch(data,1.)['log_probs']
            loss=-(out*data['advantages']*data['response_mask']).sum()/data['response_mask'].sum()
            loss.backward()
            torch.cuda.synchronize()
            elapsed=time.monotonic()-start
            assert torch.isfinite(loss)
            if iteration:
                times.append(elapsed);memory.append(torch.cuda.max_memory_allocated()/2**30);losses.append(float(loss.detach()))
            print(json.dumps({'mode':name,'iteration':iteration,'warmup':iteration==0,'seconds':elapsed}),flush=True)
        report['modes'][name]={'seconds':times,'median_seconds':statistics.median(times),
            'peak_allocated_gib':max(memory),'losses':losses}
        args.output.write_text(json.dumps(report,indent=2))
        del out,loss


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='/root/autodl-fs/tau3-core/code/checkpoints/sft-merged/new-off')
    p.add_argument('--batch',default='/root/autodl-fs/tau3-core/code/results/runs/curriculum-speed-20260912/online-smoke/update-batches/update_000001.pkl')
    p.add_argument('--row',type=int)
    p.add_argument('--iterations',type=int,default=2)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
