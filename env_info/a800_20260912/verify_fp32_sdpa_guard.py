"""Check explicit FP32 KV repetition against native mathematical GQA gradients."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
from diagnostic_attention import install_fp32_sdpa_guard
from transformers.integrations import sdpa_attention


def stats(a,b):
    x,y=a.double().flatten(),b.double().flatten()
    return {'max_abs':float((x-y).abs().max()),'relative_l2':float((x-y).norm()/x.norm().clamp_min(1e-30))}


def main(args):
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    cfg=json.loads(Path(args.config).read_text())['text_config']
    hq,hkv,dim=cfg['num_attention_heads'],cfg['num_key_value_heads'],cfg['head_dim']
    report={'kind':'FP32_GQA_vs_explicit_KV_repeat_forward_and_gradient_check','heads':[hq,hkv],
        'head_dim':dim,'cases':[]}
    for length in [128,512]:
        source=[torch.randn(1,h,length,dim,device='cuda') for h in [hq,hkv,hkv]]
        cotangent=torch.randn_like(source[0]);results=[]
        for backend in [SDPBackend.MATH,SDPBackend.EFFICIENT_ATTENTION]:
            q,k,v=[x.detach().clone().requires_grad_() for x in source]
            with sdpa_kernel(backend):
                if backend==SDPBackend.MATH:
                    out=F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
                else:
                    out=F.scaled_dot_product_attention(q,k.repeat_interleave(hq//hkv,dim=1),
                        v.repeat_interleave(hq//hkv,dim=1),is_causal=True)
                grads=torch.autograd.grad((out*cotangent).sum()/out.numel(),(q,k,v))
            results.append([out.detach(),*grads])
        case={'length':length,'comparisons':{name:stats(x,y) for name,x,y in zip(['output','q_gradient','k_gradient','v_gradient'],*results)}}
        assert all(v['relative_l2']<1e-5 for v in case['comparisons'].values())
        report['cases'].append(case)
    install_fp32_sdpa_guard()
    assert not sdpa_attention.use_gqa_in_sdpa(None,source[1])
    assert sdpa_attention.use_gqa_in_sdpa(None,source[1].bfloat16())
    report['passed']=True
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='/root/autodl-fs/tau3-core/code/checkpoints/sft-merged/new-off/config.json')
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
