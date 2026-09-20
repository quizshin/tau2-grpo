"""Read-only real-4B scoring comparisons with explicit recurrence backends."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForImageTextToText
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from verl import DataProto
from verl.utils.qwen35_padding import install_qwen35_padding_guard
from tau3_grpo.paths import CODE_ROOT
sys.path.insert(0,str(CODE_ROOT/'tests'))
from test_qwen35_compact_head import actor


def kernel(backend):
    def call(q,k,v,*,use_qk_l2norm_in_kernel=False,**kwargs):
        if backend=='native_autocast':
            return hf.torch_chunk_gated_delta_rule(q,k,v,use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,**kwargs)
        with torch.autocast('cuda',enabled=False):
            if backend=='native_fp32':
                return hf.torch_chunk_gated_delta_rule(q,k,v,use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,**kwargs)
            dtype=q.dtype
            if use_qk_l2norm_in_kernel:
                q,k=hf.l2norm(q),hf.l2norm(k)
            dtype_in=torch.float32 if backend=='fla_fp32' else torch.bfloat16
            q,k,v=[x.to(dtype_in) for x in (q,k,v)]
            kwargs={**kwargs,'beta':kwargs['beta'].to(dtype_in)}
            out,state=chunk_gated_delta_rule(q,k,v,use_qk_l2norm_in_kernel=False,**kwargs)
            return out.to(dtype),state
    return call


def compare(a,b):
    a,b=a.float(),b.float()
    delta=(a-b).abs()
    return {'max_abs':delta.max().item(),'mae':delta.mean().item(),
        'fraction_gt_004':(delta>.04).float().mean().item(),
        'fraction_gt_01':(delta>.1).float().mean().item()}


def main(args):
    if args.triton_fp32!='default':
        os.environ['TRITON_F32_DEFAULT']=args.triton_fp32
        import triton.language as tl
        importlib.import_module('fla.ops.gated_delta_rule.chunk_fwd').SOLVE_TRIL_DOT_PRECISION=tl.constexpr(args.triton_fp32)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(42)
    hf.FusedRMSNormGated=None
    model=AutoModelForImageTextToText.from_pretrained(args.model,dtype=torch.float32,
        attn_implementation='sdpa',local_files_only=True).cuda().eval()
    install_qwen35_padding_guard(model)
    modules=[m for m in model.modules() if isinstance(m,hf.Qwen3_5GatedDeltaNet)]
    if args.fp32_head:
        head=model.get_output_embeddings()
        def fp32_head(hidden):
            with torch.autocast('cuda',enabled=False):
                return torch.nn.functional.linear(hidden.float(),head.weight.float())
        head.forward=fp32_head
    data=DataProto.load_from_disk(args.batch).batch
    keys=['input_ids','attention_mask','position_ids','responses','response_mask']
    data={k:data[k][args.row:args.row+1].cuda() for k in keys}
    scored=data['response_mask'].bool().cpu()
    os.environ['VERL_QWEN35_COMPACT_HEAD']='1'
    os.environ['VERL_QWEN35_COMPACT_CHUNK']='256'
    os.environ['VERL_QWEN35_COMPACT_BACKEND']='checkpoint'
    a=actor(model)
    if args.full_fp32:
        a.param_dtype=torch.float32
    results={}
    report={'kind':'real_model_forward_only_no_RL','row':args.row,'shape':list(data['input_ids'].shape),
        'active_tokens':data['attention_mask'].sum().item(),'policy_tokens':scored.sum().item(),
        'gdn_layers':len(modules),'full_fp32':args.full_fp32,'fp32_head':args.fp32_head,'triton_fp32':args.triton_fp32,
        'modes':{},'comparisons':{}}
    captures={}
    active_positions=data['attention_mask'][0].nonzero().flatten()
    sample_positions=active_positions[::max(1,len(active_positions)//32)][:32]
    capture_left=0
    def capture(name):
        def hook(module,inputs,output):
            value=output[0] if isinstance(output,tuple) else output
            captures[name]=value[0,sample_positions-capture_left].detach().float().cpu()
        return hook
    hooks=[layer.register_forward_hook(capture(str(i))) for i,layer in enumerate(model.model.language_model.layers)] if args.capture_layers else []
    layer_outputs={}
    for backend,trim in [('native_autocast','none'),('native_fp32','none'),('fla_fp32','none'),
                         ('fla_bf16','none'),('native_fp32','experimental_both'),
                         ('fla_fp32','experimental_both'),('fla_bf16','experimental_both')]:
        name=backend+'/'+trim
        for m in modules:m.chunk_gated_delta_rule=kernel(backend)
        os.environ['VERL_QWEN35_TRIM_PADDING']=trim
        capture_left=int(active_positions[0]) if trim=='experimental_both' else 0
        captures.clear()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.monotonic()
        try:
            with torch.no_grad():out=a._forward_micro_batch(data,1.)['log_probs'].cpu()[scored]
            torch.cuda.synchronize()
            results[name]=out
            layer_outputs[name]=dict(captures)
            report['modes'][name]={'seconds':time.monotonic()-start,
                'allocated_gib':torch.cuda.max_memory_allocated()/2**30}
            if name!='native_autocast/none':
                report['comparisons'][name+' vs old']=compare(results['native_autocast/none'],out)
            if 'native_fp32/none' in results and name!='native_fp32/none':
                report['comparisons'][name+' vs fp32']=compare(results['native_fp32/none'],out)
        except Exception as e:report['modes'][name]={'error':repr(e)}
        args.output.write_text(json.dumps(report,indent=2))
        print(json.dumps({'mode':name,**report['modes'][name],
            'vs_fp32':report['comparisons'].get(name+' vs fp32')}),flush=True)
        torch.cuda.empty_cache()
    torch.save(results,args.output.with_suffix('.pt'))
    if args.capture_layers:
        torch.save(layer_outputs,args.output.with_suffix('.layers.pt'))
        report['layer_relative_l2_vs_fp32']={}
        reference=layer_outputs['native_fp32/none']
        for name,values in layer_outputs.items():
            report['layer_relative_l2_vs_fp32'][name]={k:float((reference[k]-v).norm()/reference[k].norm()) for k,v in values.items()}
        args.output.write_text(json.dumps(report,indent=2))
    for hook in hooks:hook.remove()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='/root/autodl-fs/tau3-core/code/checkpoints/sft-merged/new-off')
    p.add_argument('--batch',default='/root/autodl-fs/tau3-core/code/results/runs/rl-formal-full-20260912/acceptance/update-batches/update_000001.pkl')
    p.add_argument('--row',type=int,default=0)
    p.add_argument('--full-fp32',action='store_true')
    p.add_argument('--fp32-head',action='store_true')
    p.add_argument('--capture-layers',action='store_true')
    p.add_argument('--triton-fp32',choices=['default','ieee','tf32x3'],default='default')
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
