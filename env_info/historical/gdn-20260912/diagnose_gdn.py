"""Isolate GDN autocast errors against an independent FP64 recurrence.

No RL rollout, model weight update, or dependency mutation. Timings are cold
diagnostic timings, not a training benchmark. Each mode records failures alone.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


class MatmulDtypes(TorchDispatchMode):
    def __init__(self):
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default):
            self.counts[str((str(func), str(args[0].dtype), str(args[1].dtype), str(result.dtype)))] += 1
        return result


def oracle(query, key, value, gate, beta):
    """Direct delta rule in FP64, independent of chunk decomposition."""
    q, k, v, g, b = [x.double() for x in (query, key, value, gate, beta)]
    state = torch.zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3], device=q.device, dtype=torch.float64)
    outputs = []
    for t in range(q.shape[1]):
        state = state * g[:, t].exp()[..., None, None]
        delta = (v[:, t] - (state * k[:, t, :, :, None]).sum(-2)) * b[:, t, :, None]
        state = state + k[:, t, :, :, None] * delta[:, :, None, :]
        outputs.append((state * (q[:, t] / q.shape[-1] ** .5)[..., None]).sum(-2))
    return torch.stack(outputs, 1)


def compare(ref, candidate):
    def stats(a, b):
        a, b = a.double().flatten(), b.double().flatten()
        return {'max_abs': (a-b).abs().max().item(),
                'relative_l2': ((a-b).norm()/a.norm().clamp_min(1e-30)).item(),
                'cosine': F.cosine_similarity(a, b, dim=0).item()}
    return {'output': stats(ref['output'], candidate['output']),
            'gradients': {k: stats(ref['gradients'][k], candidate['gradients'][k]) for k in ref['gradients']}}


def run_case(length, seed):
    torch.manual_seed(seed)
    shape = (1, length, 2, 128)
    # Quantize source operands once, then hold identical values across paths.
    q, k, v = [torch.randn(shape, device='cuda').bfloat16().float() for _ in range(3)]
    q, k = [F.normalize(x, dim=-1) for x in (q, k)]
    g = -F.softplus(torch.randn(shape[:-1], device='cuda')) * .2
    beta = torch.randn(shape[:-1], device='cuda').sigmoid().bfloat16().float()
    cotangent = torch.randn_like(v)
    modes = ('oracle_fp64', 'native_fp32', 'native_autocast', 'fla_fp32', 'fla_bf16')
    outputs, report = {}, {'length': length, 'seed': seed, 'modes': {}, 'comparisons': {}}
    for mode in modes:
        tensors = [x.detach().clone().requires_grad_() for x in (q,k,v,g,beta)]
        query, key, value, gate, b = tensors
        began = time.monotonic()
        try:
            tracker = MatmulDtypes()
            with tracker, torch.autocast('cuda', dtype=torch.bfloat16, enabled=mode == 'native_autocast'):
                if mode == 'oracle_fp64':
                    output = oracle(*tensors)
                elif mode.startswith('native'):
                    output, _ = torch_chunk_gated_delta_rule(query,key,value,gate,b,use_qk_l2norm_in_kernel=False)
                else:
                    if mode == 'fla_bf16':
                        query,key,value,b = [x.bfloat16() for x in (query,key,value,b)]
                    output, _ = chunk_gated_delta_rule(query,key,value,g=gate,beta=b,
                        use_qk_l2norm_in_kernel=False,output_final_state=False)
                loss = (output.double()*cotangent.double()).sum()/output.numel()
            loss.backward()
            torch.cuda.synchronize()
            outputs[mode] = {'output': output.detach().cpu(),
                'gradients': {name:t.grad.detach().cpu() for name,t in zip(('q','k','v','g','beta'),tensors)}}
            report['modes'][mode] = {'cold_seconds':time.monotonic()-began,'matmul_dtypes':dict(tracker.counts)}
            if mode != 'oracle_fp64':
                report['comparisons'][mode] = compare(outputs['oracle_fp64'],outputs[mode])
            print(json.dumps({'length':length,'mode':mode,**report['modes'][mode],
                'vs_oracle':report['comparisons'].get(mode)}),flush=True)
        except Exception as exc:
            report['modes'][mode] = {'error':repr(exc)}
            print(json.dumps({'length':length,'mode':mode,'error':repr(exc)}),flush=True)
        finally:
            torch.cuda.empty_cache()
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lengths',type=int,nargs='+',default=[128,256,1024])
    args=parser.parse_args()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    result={'kind':'synthetic_kernel_diagnosis_not_RL','torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(),'cases':[]}
    for length in args.lengths:
        result['cases'].append(run_case(length,42))
        args.output.write_text(json.dumps(result,indent=2))
