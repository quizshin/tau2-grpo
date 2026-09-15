"""Compare external FA2 and PyTorch Flash on identical Qwen3.5 BF16 Q/K/V."""
import argparse
import json
import statistics
import time
from pathlib import Path
import torch
import flash_attn
from torch.nn.attention import sdpa_kernel, SDPBackend


def main(output,deterministic):
    torch.manual_seed(20260912)
    torch.set_num_threads(4)
    report={'torch':torch.__version__,'flash_attn':flash_attn.__version__,
            'gpu':torch.cuda.get_device_name(),'heads':[16,4],'head_dim':256,
            'dtype':'bfloat16','causal':True,'dropout':0.,'fa2_deterministic_backward':deterministic,'cases':[]}
    for length in [128,7440,17581]:
        q=torch.randn(1,length,16,256,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(1,length,4,256,device='cuda',dtype=torch.bfloat16)
        v=torch.randn_like(k)
        grad=torch.randn_like(q)*.01
        def compute(backend):
            x,y,z=[t.detach().requires_grad_(True) for t in (q,k,v)]
            if backend=='fa2':
                out=flash_attn.flash_attn_func(x,y,z,causal=True,dropout_p=0.,deterministic=deterministic)
            else:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    out=torch.nn.functional.scaled_dot_product_attention(x.transpose(1,2),y.transpose(1,2),z.transpose(1,2),
                        is_causal=True,enable_gqa=True).transpose(1,2)
            gradients=torch.autograd.grad(out,(x,y,z),grad)
            return (out.detach(),)+gradients
        outputs={}
        entry={'length':length,'timing':{}}
        for backend in ['sdpa','fa2']:
            started=time.monotonic();outputs[backend]=compute(backend);torch.cuda.synchronize()
            cold=time.monotonic()-started
            times=[]
            for _ in range(3):
                torch.cuda.synchronize();start=time.monotonic()
                values=compute(backend);torch.cuda.synchronize();times.append(time.monotonic()-start)
                del values
            entry['timing'][backend]={'cold_seconds':cold,'warm_seconds':times,'median_seconds':statistics.median(times)}
        entry['comparisons']={}
        for label,x,y in zip(['output','dq','dk','dv'],outputs['sdpa'],outputs['fa2']):
            delta=x.float()-y.float()
            entry['comparisons'][label]={'relative_l2':float(delta.norm()/x.float().norm()),'max_abs':float(delta.abs().max())}
        entry['passed']=all(x['relative_l2']<.02 for x in entry['comparisons'].values())
        report['cases'].append(entry);output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(entry),flush=True)
        del outputs,q,k,v,grad
        torch.cuda.empty_cache()
    report['passed']=all(x['passed'] for x in report['cases'])
    output.write_text(json.dumps(report,indent=2)+'\n')
    assert report['passed']

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--deterministic',action='store_true')
    args=parser.parse_args();main(args.output,args.deterministic)
