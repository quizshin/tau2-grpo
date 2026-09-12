"""Per-module GDN masking for padded singleton trajectories.

Transformers 5.5.1 skips this mask for batch size 1. A pre-hook preserves the
public forward and also runs during activation recomputation. No global state,
packing, persistent recurrent cache, or dependency files are modified.
"""


def _mask_gdn_input(module, args, kwargs):
    hidden = args[0] if args else kwargs['hidden_states']
    mask = kwargs.get('attention_mask', args[2] if len(args) > 2 else None)
    if mask is None:
        return
    if mask.ndim != 2 or tuple(mask.shape) != tuple(hidden.shape[:2]):
        raise ValueError('Qwen3.5 GDN requires a matching 2D attention mask')
    hidden = hidden * mask.to(hidden.dtype).unsqueeze(-1)
    if args:
        return (hidden, *args[1:]), kwargs
    return args, {**kwargs, 'hidden_states': hidden}


def install_qwen35_padding_guard(model):
    count = 0
    for module in model.modules():
        if type(module).__name__ == 'Qwen3_5GatedDeltaNet':
            if not hasattr(module, '_tau3_padding_guard'):
                module._tau3_padding_guard = module.register_forward_pre_hook(
                    _mask_gdn_input, with_kwargs=True)
            count += 1
    if not count:
        raise ValueError('No dense Qwen3.5 GDN modules found')
    return count


def fla_with_native_qk_norm(query, key, value, *, use_qk_l2norm_in_kernel=False, **kwargs):
    """Experimental: match native normalization and FP32 recurrence inputs."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm
    output_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query, key = l2norm(query), l2norm(key)
    # Native torch_chunk_gated_delta_rule explicitly casts all recurrence
    # operands to FP32. FLA also requires matching q/k/v/beta dtypes.
    kwargs = {**kwargs, 'beta':kwargs['beta'].float()}
    output,state = chunk_gated_delta_rule(query.float(), key.float(), value.float(),
        use_qk_l2norm_in_kernel=False, **kwargs)
    return output.to(output_dtype),state
