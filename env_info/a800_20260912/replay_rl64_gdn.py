"""Four-rank replay of all recorded RL trajectories using the real PPO actor.

Explicit FP32 compute reference; native padded/trimmed and FLA scoring, then
native/FLA trimmed PPO updates from identical weights and fresh AdamW state.
Full gradient shards and parameter deltas are compared in RAM. No checkpoint,
simulator, rollout, production dependency changes, or formal training launch.
"""
import argparse
import datetime
import importlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from omegaconf import OmegaConf
from transformers import AutoModelForImageTextToText
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from verl import DataProto
from verl.utils.fsdp_utils import get_fsdp_wrap_policy
from verl.utils.qwen35_padding import install_qwen35_padding_guard
from verl.workers.actor.dp_actor import DataParallelPPOActor
from diagnose_gdn_model import kernel
from diagnostic_attention import install_fp32_sdpa_guard,install_standard_attention_backend


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x:x.item())+'\n')


def vector_stats(reference, value):
    assert reference.shape == value.shape
    sums=torch.zeros(5,dtype=torch.float64)
    for x,y in zip(reference.reshape(-1).split(1048576),value.reshape(-1).split(1048576)):
        x,y=x.double(),y.double()
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        sums+=torch.tensor([float(x.square().sum()),float(y.square().sum()),
            float((x-y).square().sum()),float((x*y).sum()),x.numel()],dtype=torch.float64)
    return sums


def distributed_stats(sums):
    values=sums.cuda();dist.all_reduce(values);r,n,d,dot,count=values.cpu().tolist()
    return {'reference_l2':r**.5,'candidate_l2':n**.5,
        'relative_l2':(d/max(r,1e-30))**.5,'cosine':dot/max((r*n)**.5,1e-30),
        'elements':int(count)}


def score_stats(reference,value,mask):
    delta=(reference[mask]-value[mask]).abs().float()
    return {'max_abs':float(delta.max()) if delta.numel() else 0.,
        'mae':float(delta.mean()) if delta.numel() else 0.,
        'tokens':delta.numel(),'tokens_gt_004':int((delta>.04).sum())}


def run(args):
    if args.timing_only:
        assert args.reuse_native_scores is not None
        prior=json.loads((args.reuse_native_scores/'summary.json').read_text())
        assert prior['strict_numerical_gate_passed']
        assert prior.get('study','gdn')==args.study
    if args.study=='fa2':
        assert args.reuse_native_scores is not None and args.triton_fp32=='ieee'
        prior=json.loads((args.reuse_native_scores/'summary.json').read_text())
        assert prior['strict_numerical_gate_passed'],'Verify FLA before running the FA2 add-on.'
    install_fp32_sdpa_guard()
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    assert world==4
    torch.set_num_threads(4);torch.cuda.set_device(rank);torch.manual_seed(42)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    if args.triton_fp32!='default':
        os.environ['TRITON_F32_DEFAULT']=args.triton_fp32
        import triton.language as tl
        importlib.import_module('fla.ops.gated_delta_rule.chunk_fwd').SOLVE_TRIL_DOT_PRECISION=tl.constexpr(args.triton_fp32)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank),timeout=datetime.timedelta(seconds=3600))
    args.output.mkdir(parents=True,exist_ok=True)
    if rank==0:
        for filename in ['replay_rl64_gdn.py','diagnostic_attention.py','diagnose_gdn_model.py']:
            (args.output/filename).write_text((Path(__file__).parent/filename).read_text())
    report={'kind':'fixed_64_RL_trajectories_real_PPO_actor_replay_not_online_training',
        'rank':rank,'world_size':world,'precision':'FP32 outer compute, FLA FP32 inputs','triton_fp32':args.triton_fp32,
        'fp32_sdpa_explicit_kv_repeat':True,
        'param_optimizer_offload':False,'vllm_colocated':False,'batch':args.batch,
        'study':args.study,'timing_only':args.timing_only,'fa2_deterministic_backward':args.fa2_deterministic,'attention_dtype':args.attention_dtype,
        'scoring':{},'updates':{},'failures':[]}
    if args.study=='fa2':
        report['precision']=f'FP32 parameters/projections/GDN IEEE; {args.attention_dtype} standard-attention inputs and outputs cast back to FP32'
        report['mode_definitions']={'native_padded':'cached native FP32 padded numerical anchor',
            'native_trim':f'FLA IEEE plus {args.attention_dtype}-input PyTorch SDPA standard attention',
            'fla_trim':f'FLA IEEE plus external FA2 with identical {args.attention_dtype} standard-attention inputs'}
    def save():write(args.output/f'rank{rank}.json',report)
    def event(**values):print(json.dumps({'rank':rank,**values}),flush=True)
    packet=DataProto.load_from_disk(args.batch)
    assert len(packet)==64
    b=packet.batch;mask=b['response_mask'].bool();attention=b['attention_mask'].bool()
    response_length=b['responses'].shape[-1];prompt_length=b['input_ids'].shape[-1]-response_length
    assert torch.equal(b['responses'],b['input_ids'][:,-response_length:])
    assert not (mask & ~attention[:,-response_length:]).any()
    assert (b['advantages']*mask).count_nonzero()>0
    for key in ['old_log_probs','ref_log_prob','advantages']:
        assert torch.isfinite(b[key]).all(),key
    rows=[]
    for i in range(len(packet)):
        positions=attention[i].nonzero().flatten()
        selected=mask[i].nonzero().flatten()+prompt_length-1
        assert attention[i,selected].all()
        rows.append({'row':i,'task_id':str(packet.non_tensor_batch.get('task_id',['unknown']*64)[i]),
            'active_tokens':int(attention[i].sum()),'policy_tokens':int(mask[i].sum()),
            'nonpolicy_response_tokens':int((attention[i,-response_length:] & ~mask[i]).sum()),
            'nonzero_advantage_policy_tokens':int((b['advantages'][i]*mask[i]).count_nonzero()),
            'left':int(positions[0]),'right':int(positions[-1])+1})
    if rank==0:
        write(args.output/'trajectory_audit.json',{'rows':rows,'total_active_tokens':int(attention.sum()),
            'total_policy_tokens':int(mask.sum()),'nonzero_advantage_policy_tokens':int((b['advantages']*mask).count_nonzero()),
            'original_tensor_keys':list(b.keys()),'responses_match_input_suffix':True,
            'policy_positions_and_predecessors_are_active':True})
    local=packet.chunk(world)[rank]
    row_ids=list(range(rank*16,(rank+1)*16))
    local_mask=local.batch['response_mask'].bool().cpu()
    # The saved --cfg output includes launch diagnostics before its YAML block.
    config_text=Path(args.config).read_text()
    config_lines=config_text.splitlines()
    yaml_start=next(i for i,line in enumerate(config_lines) if line=='actor_rollout_ref:')
    hydra=OmegaConf.create('\n'.join(config_lines[yaml_start:]))
    source=hydra.actor_rollout_ref.actor
    cfg=OmegaConf.create({k:source.get(k,default) for k,default in {
        'ppo_epochs':1,'ppo_micro_batch_size_per_gpu':1,'use_dynamic_bsz':False,
        'use_kl_loss':True,'kl_loss_type':'low_var_kl','kl_loss_coef':.01,
        'clip_ratio':.2,'clip_ratio_low':.2,'clip_ratio_high':.2,'clip_ratio_c':3.,
        'entropy_coeff':0.,'calculate_entropy':False,'loss_agg_mode':'token-mean',
        'grad_clip':1.,'entropy_from_logits_with_chunking':False,'entropy_checkpointing':False,
        'ulysses_sequence_parallel_size':1}.items()})
    cfg.ppo_mini_batch_size=int(source.ppo_mini_batch_size)*int(hydra.actor_rollout_ref.rollout.n)//world
    cfg.policy_loss={'loss_mode':'vanilla'};cfg.global_batch_info={}
    cfg.use_remove_padding=False;cfg.use_fused_kernels=False;cfg.use_prefix_grouper=False
    cfg.use_torch_compile=False;cfg.fsdp_config={'dtype':'float32'}
    assert cfg.ppo_mini_batch_size==8 and cfg.ppo_epochs==1 and cfg.ppo_micro_batch_size_per_gpu==1
    assert not cfg.use_dynamic_bsz and cfg.entropy_coeff==0.
    local.meta_info.update(temperature=float(hydra.actor_rollout_ref.rollout.temperature),
        micro_batch_size=1,use_dynamic_bsz=False)
    report['actor_config']=OmegaConf.to_container(cfg,resolve=True)
    report['temperature']=local.meta_info['temperature']
    report['row_ids']=row_ids;save();event(stage='loading_model',rows=row_ids)
    hf.FusedRMSNormGated=None
    model=AutoModelForImageTextToText.from_pretrained(args.model,dtype=torch.float32,
        attn_implementation='sdpa',local_files_only=True)
    model.model.visual.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    install_qwen35_padding_guard(model)
    modules=[m for m in model.modules() if isinstance(m,hf.Qwen3_5GatedDeltaNet)]
    def configure_mode(mode):
        selected='fla_fp32' if mode=='fla_trim' else 'native_fp32'
        if args.study=='fa2':
            selected='fla_fp32'
            install_standard_attention_backend('fa2_bf16' if mode=='fla_trim' else 'sdpa_bf16',
                fa2_deterministic=args.fa2_deterministic,attention_dtype=getattr(torch,args.attention_dtype))
        for module in modules:module.chunk_gated_delta_rule=kernel(selected)
    wrapped=FSDP(model,device_id=torch.device('cuda',rank),
        auto_wrap_policy=get_fsdp_wrap_policy(model,is_lora=False),use_orig_params=True,
        mixed_precision=MixedPrecision(param_dtype=torch.float32,reduce_dtype=torch.float32,buffer_dtype=torch.float32))
    a=DataParallelPPOActor(cfg,wrapped)
    # Observe the first standard-attention dispatch before starting updates.
    attention_target=next(m for m in wrapped.modules() if isinstance(m,hf.Qwen3_5Attention))
    attention_profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
    attention_handles=[]
    def attention_before(module,pos,kw):
        hidden=kw.get('hidden_states',pos[0] if pos else None)
        assert hidden is not None
        report['first_standard_attention_sequence_length']=int(hidden.shape[1])
        if stage.endswith('_trim'):
            assert hidden.shape[1]==rows[row_ids[0]]['active_tokens']
        attention_profiler.start()
    def attention_after(module,pos,kw,out):
        attention_profiler.stop()
        names=[e.key for e in attention_profiler.key_averages() if 'attention' in e.key]
        report['observed_standard_attention_operators']=names
        report['first_standard_attention_inputs']=getattr(module,'_diagnostic_attention_inputs',None)
        save()
        expected='_scaled_dot_product_efficient_attention' if args.study=='gdn' else '_scaled_dot_product_flash_attention'
        assert any(expected in name for name in names),names
        for handle in attention_handles:handle.remove()
        save();event(standard_attention_operators=names)
    attention_handles.append(attention_target.register_forward_pre_hook(attention_before,with_kwargs=True))
    attention_handles.append(attention_target.register_forward_hook(attention_after,with_kwargs=True))
    os.environ.update(VERL_QWEN35_COMPACT_HEAD='1',VERL_QWEN35_COMPACT_CHUNK='256',VERL_QWEN35_COMPACT_BACKEND='checkpoint')
    original_forward=a._forward_micro_batch
    stage='';forward_index=0;last_progress=0.;training_scores=[]
    def instrumented_forward(micro_batch,*pos,**kwargs):
        nonlocal forward_index,last_progress
        out=original_forward(micro_batch,*pos,**kwargs)
        mask_here=micro_batch['response_mask'].bool()
        assert out['log_probs'].shape==mask_here.shape
        assert torch.isfinite(out['log_probs']).all()
        assert out['log_probs'][~mask_here].count_nonzero()==0
        if stage.startswith('update'):
            training_scores.append(out['log_probs'].detach().cpu())
        forward_index+=1
        if time.monotonic()-last_progress>20 or forward_index==16:
            event(stage=stage,forward_done=forward_index,of=16,row=row_ids[(forward_index-1)%16])
            last_progress=time.monotonic()
        return out
    a._forward_micro_batch=instrumented_forward
    scores={}
    for mode,trim,entropy in [('native_padded','none',False),('native_trim','experimental_both',True),('fla_trim','experimental_both',True)]:
        if args.reuse_native_scores is not None and (args.timing_only or mode=='native_padded' or (mode=='native_trim' and args.study=='gdn')):
            cached=torch.load(args.reuse_native_scores/f'scores-rank{rank}.pt',map_location='cpu',weights_only=True)
            cached_report=json.loads((args.reuse_native_scores/f'rank{rank}.json').read_text())
            assert cached_report['batch']==args.batch and cached_report['row_ids']==row_ids
            scores[mode]=cached[mode]
            record=dict(cached_report['scoring'][mode])
            record['original_seconds_local']=record['seconds_local'];record['seconds_local']=0.
            record['reused_from']=str(args.reuse_native_scores)
            report['scoring'][mode]=record;save();event(stage='score_'+mode,reused=True)
            continue
        stage='score_'+mode;forward_index=0
        configure_mode(mode)
        os.environ['VERL_QWEN35_TRIM_PADDING']=trim
        dist.barrier();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        event(stage=stage,starting=True)
        result=a.compute_log_prob(local,calculate_entropy=entropy)
        torch.cuda.synchronize();elapsed=time.monotonic()-start
        scores[mode]={k:v.detach().cpu() for k,v in result.items()}
        record={'seconds_local':elapsed,'entropy_computed':entropy,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
        if mode!='native_padded':
            record['vs_native_padded']=score_stats(scores['native_padded']['log_probs'],scores[mode]['log_probs'],local_mask)
            record['per_row']=[{'row':row_ids[i],**score_stats(scores['native_padded']['log_probs'][i],scores[mode]['log_probs'][i],local_mask[i])} for i in range(16)]
        if mode=='fla_trim':
            record['entropy_vs_native_trim']=score_stats(scores['native_trim']['entropys'],scores[mode]['entropys'],local_mask)
            record['vs_reference_trim']=score_stats(scores['native_trim']['log_probs'],scores[mode]['log_probs'],local_mask)
            record['fa2_forward_calls']=sum(getattr(m,'_diagnostic_fa2_calls',0) for m in wrapped.modules())
            if args.study=='fa2':assert record['fa2_forward_calls']>0
        record['vs_recorded_bf16_old_log_probs']=score_stats(local.batch['old_log_probs'].cpu(),scores[mode]['log_probs'],local_mask)
        report['scoring'][mode]=record;save();event(stage=stage,completed=True,seconds=elapsed)
        del result;torch.cuda.empty_cache()
    torch.save(scores,args.output/f'scores-rank{rank}.pt')
    if args.study=='fa2' and not args.timing_only:
        gates=torch.tensor([report['scoring']['fla_trim']['vs_reference_trim']['max_abs'],
            report['scoring']['native_trim']['vs_native_padded']['max_abs'],
            report['scoring']['fla_trim']['vs_native_padded']['max_abs']],device='cuda')
        dist.all_reduce(gates,op=dist.ReduceOp.MAX)
        if not bool((gates<.04).all()):
            report['stopped_before_updates']='forward numerical gate failed';save()
            if rank==0:
                write(args.output/'summary.json',{'completed':False,'study':'fa2','trajectories':64,
                    'attention_dtype':args.attention_dtype,'stopped_before_updates':'forward numerical gate failed',
                    'strict_numerical_gate_passed':False,'matched_precision_logp_max_difference':float(gates[0]),
                    'sdpa_vs_fp32_anchor_logp_max_difference':float(gates[1]),
                    'fa2_vs_fp32_anchor_logp_max_difference':float(gates[2])})
            dist.destroy_process_group();return
    initial={name:p.detach().cpu().clone() for name,p in wrapped.named_parameters()}
    reference_gradients=[];reference_deltas=[];reference_update_logps=None
    for mode in ['native_trim','fla_trim']:
        configure_mode(mode)
        # Cached scoring must not make update semantics depend on a prior loop.
        os.environ['VERL_QWEN35_TRIM_PADDING']='experimental_both'
        with torch.no_grad():
            for name,p in wrapped.named_parameters():p.copy_(initial[name].to(p.device))
        optimizer=torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad],
            lr=float(source.optim.lr),weight_decay=float(source.optim.weight_decay),foreach=False)
        a.actor_optimizer=optimizer
        stage='update_'+mode;forward_index=0;training_scores=[];update_steps=[];audit_seconds=0.
        original_step=DataParallelPPOActor._optimizer_step.__get__(a)
        def audited_step():
            nonlocal audit_seconds
            index=len(update_steps);started=time.monotonic();grads={};sums=torch.zeros(5,dtype=torch.float64)
            seen=set()
            for name,p in wrapped.named_parameters():
                if p.grad is None or p.numel()==0:continue
                assert p.requires_grad and torch.isfinite(p.grad).all(),name
                seen.add(name);value=p.grad.detach().cpu()
                if mode=='native_trim':grads[name]=value.clone()
                else:sums+=vector_stats(reference_gradients[index][name],value)
            record={'optimizer_step':index+1}
            if mode=='native_trim':
                reference_gradients.append(grads)
                count=torch.tensor(sum(v.numel() for v in grads.values()),device='cuda',dtype=torch.int64);dist.all_reduce(count)
                record['gradient_elements']=int(count)
            else:
                assert seen==set(reference_gradients[index])
                record['gradient_vs_native']=distributed_stats(sums)
            audit_seconds+=time.monotonic()-started
            torch.cuda.synchronize();began=time.monotonic()
            norm=original_step();torch.cuda.synchronize()
            record['clip_and_optimizer_seconds']=time.monotonic()-began
            record['grad_norm_before_clip']=float(norm)
            assert torch.isfinite(norm) and norm>0
            started=time.monotonic();deltas={};sums=torch.zeros(5,dtype=torch.float64);changed=0;vision_changed=0
            for name,p in wrapped.named_parameters():
                if p.numel()==0:continue
                value=p.detach().cpu();assert torch.isfinite(value).all(),name
                if not p.requires_grad:
                    vision_changed+=int((initial[name]!=value).count_nonzero());continue
                delta=value-initial[name];changed+=int(delta.count_nonzero())
                if mode=='native_trim':deltas[name]=delta
                else:sums+=vector_stats(reference_deltas[index][name],delta)
            if mode=='native_trim':reference_deltas.append(deltas)
            else:record['cumulative_parameter_delta_vs_native']=distributed_stats(sums)
            counts=torch.tensor([changed,vision_changed],device='cuda',dtype=torch.int64);dist.all_reduce(counts)
            assert counts[0]>0 and counts[1]==0
            record['changed_language_elements']=int(counts[0]);record['changed_frozen_elements']=int(counts[1])
            states=[int(v['step']) for v in optimizer.state.values() if 'step' in v]
            assert states and all(s==index+1 for s in states)
            record['adam_state_steps']=sorted(set(states))
            audit_seconds+=time.monotonic()-started
            update_steps.append(record);report['updates'][mode]={'steps':update_steps};save()
            event(stage=stage,optimizer_step=index+1,grad_norm=float(norm),gradient_comparison=record.get('gradient_vs_native'))
            return norm
        def timed_step():
            norm=original_step()
            assert torch.isfinite(norm) and norm>0
            update_steps.append({'optimizer_step':len(update_steps)+1,'grad_norm_before_clip':float(norm)})
            event(stage=stage,**update_steps[-1])
            return norm
        a._optimizer_step=timed_step if args.timing_only else audited_step
        dist.barrier();torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        started=time.monotonic();event(stage=stage,starting=True)
        metrics=a.update_policy(local)
        torch.cuda.synchronize();elapsed=time.monotonic()-started
        assert len(update_steps)==2 and forward_index==16
        computed_logps=torch.cat(training_scores)
        record={'steps':update_steps,'seconds_including_audits_local':elapsed,'audit_seconds_local':audit_seconds,
            'seconds_minus_local_audits':elapsed-audit_seconds,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'metrics':metrics,'forward_count':forward_index,'recorded_old_ref_and_advantages_reused':True}
        record['fa2_forward_calls_cumulative']=sum(getattr(m,'_diagnostic_fa2_calls',0) for m in wrapped.modules())
        if mode=='native_trim':reference_update_logps=computed_logps
        else:record['training_log_probs_vs_native']=score_stats(reference_update_logps,computed_logps,local_mask)
        report['updates'][mode]=record;save();event(stage=stage,completed=True,seconds_excluding_local_audits=elapsed-audit_seconds)
        torch.save(computed_logps,args.output/f'update-logps-{mode}-rank{rank}.pt')
        a.actor_optimizer=None;del optimizer;wrapped.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    report['completed']=True;save()
    dist.barrier()
    if rank==0:
        reports=[json.loads((args.output/f'rank{i}.json').read_text()) for i in range(world)]
        summary={'completed':True,'study':args.study,'trajectories':64,'world_size':world,'updates_per_backend':2,
            'all_frozen_parameters_unchanged':True,'online_rollout_tested':False,'production_bf16_equivalence_claimed':False,
            'scoring':{},'updates':{}}
        for mode in scores:
            entries=[r['scoring'][mode] for r in reports]
            summary['scoring'][mode]={'max_rank_seconds':max(x['seconds_local'] for x in entries)}
            if mode!='native_padded':
                summary['scoring'][mode].update(max_logp_difference=max(x['vs_native_padded']['max_abs'] for x in entries),
                    tokens_gt_004=sum(x['vs_native_padded']['tokens_gt_004'] for x in entries))
        for mode in ['native_trim','fla_trim']:
            entries=[r['updates'][mode] for r in reports]
            summary['updates'][mode]={'max_rank_seconds_minus_local_audits':max(x['seconds_minus_local_audits'] for x in entries),
                'max_peak_allocated_gib':max(x['peak_allocated_gib'] for x in entries),'steps':entries[0]['steps']}
        if args.timing_only:
            summary['timing_only']=True
            summary['all_frozen_parameters_unchanged']=None
            summary['correctness_evidence']=str(args.reuse_native_scores)
            summary['kernel_cache']='reused from completed correctness run; no CPU gradient/delta audits'
            summary['candidate_speedup']=summary['updates']['native_trim']['max_rank_seconds_minus_local_audits']/summary['updates']['fla_trim']['max_rank_seconds_minus_local_audits']
            write(args.output/'summary.json',summary);event(summary=summary)
            dist.destroy_process_group()
            return
        summary['strict_numerical_gate_passed']=(summary['scoring']['native_trim']['max_logp_difference']<.04
            and summary['scoring']['fla_trim']['max_logp_difference']<.04
            and all(x['gradient_vs_native']['relative_l2']<.03 and x['gradient_vs_native']['cosine']>.999 for x in summary['updates']['fla_trim']['steps']))
        summary['update_logp_max_difference']=max(r['updates']['fla_trim']['training_log_probs_vs_native']['max_abs'] for r in reports)
        summary['high_precision_anchor_logp_gate_passed']=(summary['scoring']['native_trim']['max_logp_difference']<.04
            and summary['scoring']['fla_trim']['max_logp_difference']<.04)
        if args.study=='fa2':
            summary['mode_definitions']=reports[0]['mode_definitions']
            summary['matched_precision_logp_max_difference']=max(r['scoring']['fla_trim']['vs_reference_trim']['max_abs'] for r in reports)
            summary['strict_numerical_gate_passed']=(summary['matched_precision_logp_max_difference']<.04
                and all(s['gradient_vs_native']['relative_l2']<.03 and s['gradient_vs_native']['cosine']>.999 for s in summary['updates']['fla_trim']['steps']))
            summary['fa2_forward_calls_per_rank']=[r['updates']['fla_trim']['fa2_forward_calls_cumulative'] for r in reports]
        write(args.output/'summary.json',summary);event(summary=summary)
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='/root/autodl-fs/tau3-core/code/checkpoints/sft-merged/new-off')
    p.add_argument('--batch',default='/root/autodl-fs/tau3-core/code/results/runs/curriculum-speed-20260912/online-smoke/update-batches/update_000001.pkl')
    p.add_argument('--config',default='/root/autodl-fs/tau3-core/code/results/runs/curriculum-speed-20260912/smoke.hydra.yaml')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--triton-fp32',choices=['default','ieee'],default='default')
    p.add_argument('--reuse-native-scores',type=Path)
    p.add_argument('--study',choices=['gdn','fa2'],default='gdn')
    p.add_argument('--timing-only',action='store_true',help='Requires completed correctness evidence; replay without CPU gradient/delta audits using existing kernel cache.')
    p.add_argument('--fa2-deterministic',action='store_true',help='Use optional deterministic FA2 backward; default uses its standard faster backward.')
    p.add_argument('--attention-dtype',choices=['bfloat16','float16'],default='bfloat16')
    run(p.parse_args())
