"""Observe the actual standard-attention operator in one real BF16 forward."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from verl import DataProto
from verl.utils.qwen35_padding import install_qwen35_padding_guard
from diagnose_gdn_model import actor,kernel
from diagnostic_attention import install_fp32_sdpa_guard,install_standard_attention_backend


def main(args):
    if args.fp32_sdpa_guard:install_fp32_sdpa_guard()
    torch.set_num_threads(4);torch.manual_seed(42)
    hf.FusedRMSNormGated=None
    model=AutoModelForImageTextToText.from_pretrained(args.model,dtype=torch.float32,
        attn_implementation='sdpa',local_files_only=True).cuda().eval()
    install_qwen35_padding_guard(model)
    for module in model.modules():
        if isinstance(module,hf.Qwen3_5GatedDeltaNet):module.chunk_gated_delta_rule=kernel('native_autocast')
    data=DataProto.load_from_disk(args.batch).batch
    keys=['input_ids','attention_mask','position_ids','responses','response_mask']
    data={k:data[k][args.row:args.row+1].cuda() for k in keys}
    target=next(m for m in model.modules() if isinstance(m,hf.Qwen3_5Attention))
    os.environ.update(VERL_QWEN35_COMPACT_HEAD='1',VERL_QWEN35_COMPACT_CHUNK='256',
        VERL_QWEN35_COMPACT_BACKEND='checkpoint',VERL_QWEN35_TRIM_PADDING='none')
    report={'row':args.row,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'flash_attn_package_importable':importlib.util.find_spec('flash_attn') is not None,
        'attn_implementation':model.config.text_config._attn_implementation,
        'full_fp32':args.full_fp32,'fp32_sdpa_guard':args.fp32_sdpa_guard,
        'scope':'First standard-attention layer of real padded/trimmed forwards; no timing benchmark',
        'modes':{}}
    def before(module,pos,kw):
        mask=kw.get('attention_mask')
        mode_report['attention_mask_shape']=list(mask.shape) if isinstance(mask,torch.Tensor) else str(type(mask))
        mode_report['attention_mask_dtype']=str(mask.dtype) if isinstance(mask,torch.Tensor) else None
        profiler.start()
    def after(module,pos,kw,out):profiler.stop()
    pre=target.register_forward_pre_hook(before,with_kwargs=True)
    post=target.register_forward_hook(after,with_kwargs=True)
    for trim in ['experimental_both']:
        install_standard_attention_backend('sdpa_bf16')
        os.environ['VERL_QWEN35_TRIM_PADDING']=trim
        mode_report={}
        profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
        a=actor(model)
        if args.full_fp32:a.param_dtype=torch.float32
        try:
            with torch.no_grad():a._forward_micro_batch(data,1.)
        finally:
            print('DIAGNOSTIC_INPUTS',getattr(target,'_diagnostic_attention_inputs',None),flush=True)
        torch.cuda.synchronize()
        mode_report['attention_operators']=[{'name':e.key,'calls':e.count} for e in profiler.key_averages()
            if any(word in e.key for word in ['scaled_dot','flash_attention','efficient_attention','cudnn_attention'])]
        report['modes'][trim]=mode_report
        print(json.dumps({'trim':trim,**mode_report}),flush=True)
    pre.remove();post.remove()
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='/root/autodl-fs/tau3-core/code/checkpoints/sft-merged/new-off')
    p.add_argument('--batch',default='/root/autodl-fs/tau3-core/code/results/runs/curriculum-speed-20260912/online-smoke/update-batches/update_000001.pkl')
    p.add_argument('--row',type=int,default=16)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--full-fp32',action='store_true')
    p.add_argument('--fp32-sdpa-guard',action='store_true')
    main(p.parse_args())
