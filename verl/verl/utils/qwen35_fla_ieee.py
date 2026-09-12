"""Opt-in Qwen3.5 FLA path validated against full FP32 RL gradients.

Restricted to independent singleton text trajectories. No sequence packing,
state reuse, external FA2, or model/optimizer state-dict changes.
"""
import importlib
import json
import os


def prepare_fla_ieee_runtime():
    import torch
    import triton.language as tl
    from transformers.integrations import sdpa_attention
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    os.environ['TRITON_F32_DEFAULT']='ieee'
    importlib.import_module('fla.ops.gated_delta_rule.chunk_fwd').SOLVE_TRIL_DOT_PRECISION=tl.constexpr('ieee')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    # Must run before construction to retain the native norm and state layout.
    hf.FusedRMSNormGated=None
    current=sdpa_attention.use_gqa_in_sdpa
    if not getattr(current,'_tau3_fp32_guard',False):
        def guarded(mask,key):
            return False if key.dtype==torch.float32 else current(mask,key)
        guarded._tau3_fp32_guard=True
        sdpa_attention.use_gqa_in_sdpa=guarded


def install_fla_ieee(model):
    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    assert model.config.model_type=='qwen3_5'
    assert os.environ.get('VERL_QWEN35_COMPACT_HEAD')=='1'
    assert os.environ.get('VERL_QWEN35_TRIM_PADDING')=='experimental_both'
    modules=[m for m in model.modules() if isinstance(m,hf.Qwen3_5GatedDeltaNet)]
    assert modules and all(isinstance(m.norm,hf.Qwen3_5RMSNormGated) for m in modules)
    calls={'count':0,'phases':set()}
    def kernel(q,k,v,*,use_qk_l2norm_in_kernel=False,**kwargs):
        assert q.shape[0]==1, 'FLA IEEE online probe requires independent singleton trajectories'
        assert q.dtype==k.dtype==v.dtype==torch.float32, 'FP32 compute must be explicit in FSDP and actor config'
        assert kwargs.get('initial_state') is None, 'No persistent recurrent cache in PPO scoring or updates'
        phase='backprop_enabled' if torch.is_grad_enabled() else 'no_grad'
        if phase not in calls['phases']:
            print(json.dumps({'tau3_fla_ieee':'first_kernel_call','pid':os.getpid(),
                'phase':phase,'shape':list(q.shape),'dtype':str(q.dtype),'triton_precision':'ieee'}),flush=True)
            calls['phases'].add(phase)
        calls['count']+=1
        with torch.autocast('cuda',enabled=False):
            if use_qk_l2norm_in_kernel:q,k=hf.l2norm(q),hf.l2norm(k)
            kwargs={**kwargs,'beta':kwargs['beta'].float()}
            output,state=chunk_gated_delta_rule(q.float(),k.float(),v.float(),
                use_qk_l2norm_in_kernel=False,**kwargs)
            return output,state
    for module in modules:module.chunk_gated_delta_rule=kernel
    model._tau3_fla_ieee_calls=calls
    print(json.dumps({'tau3_fla_ieee':'installed','pid':os.getpid(),'gdn_layers':len(modules),
        'native_norm':True,'fp32_sdpa_explicit_kv_repeat':True}),flush=True)
    return len(modules)
