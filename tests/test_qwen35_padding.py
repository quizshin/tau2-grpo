"""Padding invariance against real unpadded inputs, with full gradients."""
import copy

import pytest
import torch

from test_qwen35 import tiny_model
from verl.utils.qwen35_padding import install_qwen35_padding_guard


@pytest.mark.parametrize('batch_size',[1,2])
@pytest.mark.parametrize('tied',[False,True])
def test_guard_matches_unpadded_full_gradients(batch_size,tied):
    if not torch.cuda.is_available():
        pytest.skip('GPU numerical test')
    torch.manual_seed(43)
    base=tiny_model(tied).cuda()
    base.model.visual.requires_grad_(False)
    with torch.no_grad():
        base.get_input_embeddings().weight[0].normal_(std=.1)
    models=[base,copy.deepcopy(base)]
    assert install_qwen35_padding_guard(models[1])==1
    assert install_qwen35_padding_guard(models[1])==1
    real=torch.arange(2,10,device='cuda')[None].repeat(batch_size,1)
    if batch_size==2:
        real[1]+=8
    ids=torch.zeros((batch_size,192),device='cuda',dtype=torch.long)
    attention=torch.zeros_like(ids)
    lefts=[65,68][:batch_size]
    for row,left in enumerate(lefts):
        ids[row,left:left+8]=real[row]
        attention[row,left:left+8]=1
    results=[]
    for index,model in enumerate(models):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if index==0:
                logits=model(input_ids=real,use_cache=False).logits[:,:-1]
            else:
                output=model(input_ids=ids,attention_mask=attention,
                    position_ids=(attention.cumsum(-1)-1).clamp_min(0),use_cache=False).logits
                logits=torch.stack([output[row,left:left+7] for row,left in enumerate(lefts)])
            logp=logits.float().log_softmax(-1).gather(-1,real[:,1:,None]).squeeze(-1)
            loss=-logp.mean()
        loss.backward()
        results.append(logp.detach())
    torch.testing.assert_close(*results,atol=.04,rtol=.01)
    for (name,p),(_,q) in zip(models[0].named_parameters(),models[1].named_parameters()):
        assert (p.grad is None)==(q.grad is None),name
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,atol=.01,rtol=.04)
