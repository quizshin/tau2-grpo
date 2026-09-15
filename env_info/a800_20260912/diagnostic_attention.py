"""Process-local FP32 SDPA routing for numerical reference runs.

HF enables native GQA when the causal mask is omitted. PyTorch's efficient
SDPA kernel does not accept native GQA, while its Flash kernel rejects FP32.
Explicitly repeat KV heads for FP32 to keep the efficient kernel eligible.
BF16/FP16 dispatch and all installed dependency files remain unchanged.
"""
import torch
from transformers.integrations import sdpa_attention
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def install_fp32_sdpa_guard():
    current=sdpa_attention.use_gqa_in_sdpa
    if getattr(current,'_diagnostic_fp32_guard',False):return
    def guarded(attention_mask,key):
        return False if key.dtype==torch.float32 else current(attention_mask,key)
    guarded._diagnostic_fp32_guard=True
    sdpa_attention.use_gqa_in_sdpa=guarded


def install_standard_attention_backend(backend,*,fa2_deterministic=False,attention_dtype=torch.bfloat16):
    """Matched BF16 attention inputs; all projections and GDN stay FP32.

    FA2 is imported only when explicitly selected after installation. These
    wrappers support the cropped text-only, independent-sequence diagnostics.
    """
    if backend=='native':
        ALL_ATTENTION_FUNCTIONS.register('sdpa',sdpa_attention.sdpa_attention_forward)
        return
    if backend not in ['sdpa_bf16','fa2_bf16']:raise ValueError(backend)
    def forward(module,query,key,value,attention_mask,dropout=0.,scaling=None,**kwargs):
        original_dtype=query.dtype
        q,k,v=[x.to(attention_dtype) for x in (query,key,value)]
        module._diagnostic_attention_inputs={'backend':backend,'q_dtype':str(q.dtype),
            'q_shape':list(q.shape),'q_stride':list(q.stride()),'k_stride':list(k.stride()),'v_stride':list(v.stride()),
            'mask_shape':None if attention_mask is None else list(attention_mask.shape),
            'flash_enabled':torch.backends.cuda.flash_sdp_enabled()}
        if backend=='sdpa_bf16':
            from torch.nn.attention import sdpa_kernel,SDPBackend
            # The outer actor's FP32 autocast would otherwise upcast BF16
            # SDPA arguments again before dispatch, forcing math attention.
            with torch.autocast('cuda',enabled=False),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                result,_=sdpa_attention.sdpa_attention_forward(module,q,k,v,attention_mask,
                    dropout=dropout,scaling=scaling,**kwargs)
        else:
            from flash_attn import flash_attn_func
            if attention_mask is not None:
                raise ValueError('FA2 diagnostic requires cropped contiguous sequences with no explicit attention mask.')
            causal=kwargs.get('is_causal',getattr(module,'is_causal',True))
            if causal is None:causal=getattr(module,'is_causal',True)
            with torch.autocast('cuda',enabled=False):
                result=flash_attn_func(q.transpose(1,2).contiguous(),k.transpose(1,2).contiguous(),
                    v.transpose(1,2).contiguous(),dropout_p=dropout,softmax_scale=scaling,
                    causal=bool(causal and q.shape[2]>1),deterministic=fa2_deterministic)
            module._diagnostic_fa2_calls=getattr(module,'_diagnostic_fa2_calls',0)+1
        return result.to(original_dtype),None
    ALL_ATTENTION_FUNCTIONS.register('sdpa',forward)
