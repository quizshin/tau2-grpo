"""Isolated four-rank full-parameter PPO replay; never starts online training.

Uses recorded real tokens/masks/advantages. No new reward claim or checkpoint.
Peak GPU memory excludes colocated vLLM/reference replicas.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from transformers import AutoModelForImageTextToText

from tau3_grpo.paths import CODE_ROOT
from verl import DataProto
from verl.utils.fsdp_utils import get_fsdp_wrap_policy

sys.path.insert(0, str(CODE_ROOT / 'tests'))
from test_qwen35_compact_head import actor, batch as tiny_batch
from test_qwen35 import tiny_model


def wrap(model, rank):
    model.model.visual.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return FSDP(model, device_id=torch.device('cuda', rank),
        auto_wrap_policy=get_fsdp_wrap_policy(model, is_lora=False), use_orig_params=True,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32, buffer_dtype=torch.float32))


def run(args):
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', device_id=torch.device('cuda', rank),
        timeout=datetime.timedelta(seconds=900))
    torch.manual_seed(42)
    # Installing FLA would also replace RMSNorm automatically. Keep the native
    # norm so --kernel=fla changes only the Gated Delta Rule kernel.
    from transformers.models.qwen3_5 import modeling_qwen3_5
    modeling_qwen3_5.FusedRMSNormGated = None
    model = tiny_model(args.tied) if args.tiny else AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation='sdpa', local_files_only=True)
    if args.kernel == 'fla':
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        count = 0
        for module in model.modules():
            if hasattr(module, 'chunk_gated_delta_rule'):
                module.chunk_gated_delta_rule = chunk_gated_delta_rule
                count += 1
        assert count
    wrapped = wrap(model, rank)
    a = actor(wrapped)
    if args.tiny:
        data = tiny_batch()
        micros = [{k:v[m:m+1] for k,v in data.items()} for m in range(2)]
        for m, micro in enumerate(micros):
            if rank == m:
                micro['response_mask'].zero_()
            micro['advantages'] = torch.ones_like(micro['response_mask'], dtype=torch.float32)
    else:
        data = DataProto.load_from_disk(args.batch).batch
        assert len(data) == 8 and dist.get_world_size() == 4
        keys = ['input_ids', 'attention_mask', 'position_ids', 'responses', 'response_mask', 'advantages']
        micros = [{k:data[k][i:i+1].cuda() for k in keys} for i in (rank, rank+4)]
    mask_sum = sum(m['response_mask'].sum() for m in micros).float()
    dist.all_reduce(mask_sum)
    assert mask_sum > 0
    optimizer = torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad],
        lr=1e-6, foreach=False)
    report = {'mode':args.mode, 'kernel':args.kernel, 'rank':rank, 'world_size':dist.get_world_size(),
        'trim_padding':os.environ.get('VERL_QWEN35_TRIM_PADDING','none'),
        'normalization':'native',
        'tiny':args.tiny, 'tied':args.tied, 'kind':'recorded_batch_replay_not_online_RL',
        'vllm_colocated':False, 'input_shapes':[list(m['input_ids'].shape) for m in micros],
        'policy_tokens':[int(m['response_mask'].sum()) for m in micros], 'timings':{}, 'peaks':{}}

    def start():
        dist.barrier();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        return time.perf_counter()

    def finish(name, began):
        torch.cuda.synchronize()
        seconds = torch.tensor(time.perf_counter()-began, device='cuda')
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        report['timings'][name] = seconds.item()
        report['peaks'][name] = {'allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'reserved_gib':torch.cuda.max_memory_reserved()/2**30}
        if rank == 0:
            print(json.dumps({'phase':name,'seconds':seconds.item(),'peak':report['peaks'][name]}),flush=True)

    os.environ['VERL_QWEN35_COMPACT_HEAD'] = str(int(args.mode == 'compact'))
    os.environ['VERL_QWEN35_COMPACT_CHUNK'] = str(args.chunk)
    wrapped.eval()
    old, refs = [], []
    t = start()
    with torch.no_grad():
        for micro in micros:
            old.append(a._forward_micro_batch(micro, 1., calculate_entropy=True)['log_probs'].detach())
    finish('old_log_prob', t)
    torch.cuda.empty_cache()
    t = start()
    with torch.no_grad():
        for micro in micros:
            refs.append(a._forward_micro_batch(micro, 1.)['log_probs'].detach())
    finish('ref_log_prob_same_initial_weights', t)
    torch.cuda.empty_cache()
    wrapped.train()
    t = start()
    total_loss, scored = 0., []
    for micro, previous, ref in zip(micros, old, refs):
        logp = a._forward_micro_batch(micro, 1.)['log_probs']
        mask, adv = micro['response_mask'], micro['advantages']
        ratio = torch.exp((logp-previous).clamp(-20,20))
        pg = torch.maximum(-adv*ratio, -adv*ratio.clamp(.8,1.2))
        kl = (ref-logp).clamp(-20,20)
        loss = ((pg + .01*(torch.exp(kl)-kl-1))*mask).sum()*dist.get_world_size()/mask_sum
        loss.backward()
        total_loss += float(loss.detach())
        scored.append(logp.detach().cpu()*mask.cpu())
    norm = wrapped.clip_grad_norm_(1.)
    assert torch.isfinite(norm) and norm > 0
    report.update(loss=total_loss, grad_norm=float(norm))
    finish('forward_backward', t)
    # Small deterministic samples allow real-model numerical comparisons without
    # exporting complete gradients or model weights.
    samples = {}
    tracked = None
    for name, p in wrapped.named_parameters():
        if p.grad is not None and p.numel():
            assert torch.isfinite(p.grad).all(), name
            stride = max(1, p.numel()//256)
            samples[name] = p.grad.detach().flatten()[::stride][:256].cpu()
            if tracked is None and torch.count_nonzero(p.grad):
                tracked = p
    before = tracked.detach().flatten()[:4096].cpu().clone() if tracked is not None else None
    t = start();optimizer.step();finish('optimizer',t)
    changed = torch.tensor(int(tracked is not None and not torch.equal(before,
        tracked.detach().flatten()[:4096].cpu())), device='cuda')
    dist.all_reduce(changed)
    assert changed > 0
    report['optimizer_changed_ranks'] = int(changed)
    out = Path(args.output);out.mkdir(parents=True,exist_ok=True)
    torch.save({'gradient_samples':samples,'log_probs':scored},out/f'rank{rank}.pt')
    (out/f'rank{rank}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['baseline','compact'],required=True)
    p.add_argument('--kernel',choices=['native','fla'],default='native')
    p.add_argument('--model',default='/root/autodl-fs/tau3-core-20260912/checkpoints/sft-merged/new-off')
    p.add_argument('--batch',default='/root/autodl-fs/tau3-core-20260912/runs/rl-formal-full-20260912/acceptance/update-batches/update_000001.pkl')
    p.add_argument('--output',required=True)
    p.add_argument('--tiny',action='store_true')
    p.add_argument('--tied',action='store_true')
    p.add_argument('--chunk',type=int,default=256)
    run(p.parse_args())
