"""Recheck all frozen RL policy tokens with IEEE FLA against saved FSDP scores."""
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
from diagnose_gdn_model import actor,kernel,compare


def main(args):
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    os.environ['TRITON_F32_DEFAULT']='ieee'
    import triton.language as tl
    importlib.import_module('fla.ops.gated_delta_rule.chunk_fwd').SOLVE_TRIL_DOT_PRECISION=tl.constexpr('ieee')
    refs=[torch.load(args.reference/f'scores-rank{i}.pt',map_location='cpu',weights_only=True) for i in range(4)]
    padded=torch.cat([r['native_padded']['log_probs'] for r in refs])
    trimmed=torch.cat([r['native_trim']['log_probs'] for r in refs])
    packet=DataProto.load_from_disk(args.batch);batch=packet.batch
    assert len(batch)==64
    hf.FusedRMSNormGated=None
    model=AutoModelForImageTextToText.from_pretrained(args.model,dtype=torch.float32,
        attn_implementation='sdpa',local_files_only=True).cuda().eval()
    install_qwen35_padding_guard(model)
    modules=[m for m in model.modules() if isinstance(m,hf.Qwen3_5GatedDeltaNet)]
    a=actor(model);a.param_dtype=torch.float32
    os.environ.update(VERL_QWEN35_COMPACT_HEAD='1',VERL_QWEN35_COMPACT_CHUNK='256',
        VERL_QWEN35_COMPACT_BACKEND='checkpoint',VERL_QWEN35_TRIM_PADDING='experimental_both')
    priority=[12,21,39,6,56];order=priority+[i for i in range(64) if i not in priority]
    report={'kind':'all_64_real_RL_trajectory_forward_IEEE_FLA_check','triton_fp32':'ieee',
        'reference':str(args.reference),'rows':[]}
    all_scores=torch.zeros_like(padded)
    keys=['input_ids','attention_mask','position_ids','responses','response_mask']
    for row in order:
        data={k:batch[k][row:row+1].cuda() for k in keys};mask=data['response_mask'].bool().cpu()[0]
        entry={'row':row,'active_tokens':int(data['attention_mask'].sum()),'policy_tokens':int(mask.sum())}
        if row in priority:
            for module in modules:module.chunk_gated_delta_rule=kernel('native_fp32')
            with torch.no_grad():native=a._forward_micro_batch(data,1.)['log_probs'][0].cpu()
            entry['native_single_gpu_vs_saved_fsdp']=compare(trimmed[row][mask],native[mask])
            assert entry['native_single_gpu_vs_saved_fsdp']['max_abs']<.001
        for module in modules:module.chunk_gated_delta_rule=kernel('fla_fp32')
        torch.cuda.synchronize();began=time.monotonic()
        with torch.no_grad():value=a._forward_micro_batch(data,1.)['log_probs'][0].cpu()
        torch.cuda.synchronize();entry['cold_forward_seconds']=time.monotonic()-began
        assert torch.isfinite(value).all() and value[~mask].count_nonzero()==0
        all_scores[row]=value
        entry['vs_native_padded']=compare(padded[row][mask],value[mask])
        entry['vs_native_trim']=compare(trimmed[row][mask],value[mask])
        report['rows'].append(entry)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(entry),flush=True)
    report['completed']=True
    report['max_logp_difference']=max(r['vs_native_padded']['max_abs'] for r in report['rows'])
    report['passed']=report['max_logp_difference']<.04
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    torch.save(all_scores,args.output.with_suffix('.pt'))
    print(json.dumps({'completed':True,'max_logp_difference':report['max_logp_difference'],'passed':report['passed']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='/root/autodl-fs/tau3-core/code/checkpoints/sft-merged/new-off')
    p.add_argument('--batch',default='/root/autodl-fs/tau3-core/code/results/runs/curriculum-speed-20260912/online-smoke/update-batches/update_000001.pkl')
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
