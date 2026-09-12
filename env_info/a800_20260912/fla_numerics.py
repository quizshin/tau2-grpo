"""Optional FLA forward/backward comparison against native PyTorch on A800."""
import copy
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from transformers.models.qwen3_5 import modeling_qwen3_5 as native
from tau3_grpo.paths import CODE_ROOT

sys.path.insert(0, str(CODE_ROOT/'tests'))
from test_qwen35 import tiny_model
from test_qwen35_compact_head import actor, batch


def difference(a,b):
    a,b=a.float().flatten(),b.float().flatten()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return {'relative_l2':float((a-b).norm()/a.norm().clamp_min(1e-12)),
        'max_absolute':float((a-b).abs().max()),'cosine':float(F.cosine_similarity(a,b,dim=0))}


report={'kernel_checks':[]}
torch.manual_seed(42)
for length in (63,257,2049):
    source=[torch.randn(1,length,2,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    source += [-torch.rand(1,length,2,device='cuda')*.2,
               torch.rand(1,length,2,device='cuda',dtype=torch.bfloat16)]
    results=[]
    for kernel in (native.torch_chunk_gated_delta_rule,chunk_gated_delta_rule):
        values=[x.detach().clone().requires_grad_() for x in source]
        out,_=kernel(*values[:3],g=values[3],beta=values[4],use_qk_l2norm_in_kernel=True,
            output_final_state=False)
        out.float().square().mean().backward()
        results.append((out.detach(),[x.grad.detach() for x in values]))
    item={'length':length,'output':difference(results[0][0],results[1][0]),
        'gradients':[difference(a,b) for a,b in zip(results[0][1],results[1][1])]}
    assert item['output']['relative_l2']<.015,item
    assert all(g['relative_l2']<.03 and g['cosine']>.999 for g in item['gradients']),item
    report['kernel_checks'].append(item)
    print(json.dumps(item),flush=True)

model=tiny_model(False).cuda()
model.model.visual.requires_grad_(False)
models=[model,copy.deepcopy(model)]
data=batch();results=[]
for i,model in enumerate(models):
    for m in model.modules():
        if hasattr(m,'chunk_gated_delta_rule'):
            m.chunk_gated_delta_rule=(native.torch_chunk_gated_delta_rule,chunk_gated_delta_rule)[i]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    result=actor(model)._forward_micro_batch(data,1.)['log_probs']
    (-(result*data['response_mask']).sum()/data['response_mask'].sum()).backward()
    results.append((result.detach()*data['response_mask'],torch.cat([p.grad.flatten()
        for p in model.parameters() if p.grad is not None])))
report['full_parameter_output']=difference(results[0][0],results[1][0])
report['full_parameter_gradients']=difference(results[0][1],results[1][1])
assert report['full_parameter_output']['max_absolute']<.04
assert report['full_parameter_gradients']['relative_l2']<.03
assert report['full_parameter_gradients']['cosine']>.999
report['passed']=True
print(json.dumps(report),flush=True)
Path(sys.argv[1]).write_text(json.dumps(report,indent=2))
