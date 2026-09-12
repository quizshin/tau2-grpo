"""Opt-in Qwen3.5 PPO statistics with a trainable, chunked vocabulary head.

Keep native attention and recurrent state. Vocabulary projections outside
response_mask are omitted. An independent opt-in trims common right padding;
real dialogue tokens are never removed or packed. The fused backend and padding
trims are retained for isolated comparisons only: they failed real-model gates.
The validated candidate uses backend=checkpoint and trim_padding=none.
"""
from contextlib import contextmanager
import os

import torch

from verl.utils.experimental.torch_functional import FusedLinearForPPO
from verl.utils.torch_functional import logprobs_from_logits, entropy_from_logits
from torch.utils.checkpoint import checkpoint


@contextmanager
def compact_head(model, labels, temperature, chunk_size, calculate_entropy):
    head = model.get_output_embeddings()
    if not isinstance(head, torch.nn.Linear) or head.bias is not None:
        raise ValueError("Compact PPO head requires the native bias-free linear head")
    original = head.forward
    had_override = "forward" in head.__dict__
    kernel = FusedLinearForPPO(chunk_size=chunk_size)

    def statistics(hidden, targets):
        logits = original(hidden)
        logits.div_(temperature)
        logp = logprobs_from_logits(logits, targets)
        if calculate_entropy:
            return torch.stack((logp, entropy_from_logits(logits)), dim=-1)
        return logp.unsqueeze(-1)

    def forward(hidden):
        if hidden.shape[:2] != labels.shape:
            raise ValueError("Policy labels and projected hidden states are misaligned")
        # Outside FSDP the master weight may be FP32. Explicitly cast so the
        # custom backward receives matched dtypes and casts gradients back.
        if os.environ.get("VERL_QWEN35_COMPACT_BACKEND", "checkpoint") == "fused":
            logp, entropy = kernel(hidden, head.weight.to(hidden.dtype), labels, temperature)
            return torch.stack((logp, entropy), dim=-1)
        chunks = []
        for start in range(0, hidden.shape[1], chunk_size):
            part, targets = hidden[:,start:start+chunk_size], labels[:,start:start+chunk_size]
            if torch.is_grad_enabled() and (part.requires_grad or head.weight.requires_grad):
                values = checkpoint(statistics, part, targets, use_reentrant=False, preserve_rng_state=False)
            else:
                values = statistics(part, targets)
            chunks.append(values)
        return torch.cat(chunks, dim=1)

    head.forward = forward
    try:
        yield
    finally:
        if had_override:
            head.forward = original
        else:
            del head.forward


def compact_forward(actor, batch, temperature, calculate_entropy, chunk_size):
    if (actor.actor_module.config.model_type != "qwen3_5" or actor.use_remove_padding
            or actor.use_ulysses_sp or actor.use_fused_kernels or actor.use_prefix_grouper
            or actor.config.get("calculate_sum_pi_squared", False)
            or batch.get("multi_modal_inputs")):
        raise ValueError("Compact head requires native padded text-only Qwen3.5 PPO")
    ids, responses, mask = batch["input_ids"], batch["responses"], batch.get("response_mask")
    if mask is None or mask.shape != responses.shape or chunk_size < 1:
        raise ValueError("Complete response_mask and a positive chunk size are required")
    prompt_length = ids.shape[-1] - responses.shape[-1]
    if prompt_length < 1:
        raise ValueError("A nonempty prompt is required")
    columns = mask.bool().any(0).nonzero(as_tuple=True)[0]
    if columns.numel() == 0:
        # Empty local ranks still participate in FSDP forward/backward.
        columns = torch.zeros(1, dtype=torch.long, device=ids.device)
    positions = batch["position_ids"]
    if positions.dim() == 3:
        positions = positions.transpose(0, 1)
    attention = batch["attention_mask"]
    selected = columns + prompt_length - 1
    trim = os.environ.get("VERL_QWEN35_TRIM_PADDING", "none")
    if trim not in ("none", "right", "experimental_both"):
        raise ValueError("Unknown Qwen3.5 padding trim mode")
    if trim != "none":
        active = attention.bool().any(0).nonzero(as_tuple=True)[0]
        if active.numel():
            # Preserve all left padding: Transformers 5.5.1's GDN masking
            # skips singleton batches, so deleting it changes recurrent state.
            left = int(active[0]) if trim == "experimental_both" else 0
            # Preserve the last native 64-token GDN chunk as well.
            right = (int(active[-1]) + 1 if trim == "experimental_both" else
                     min(ids.shape[-1], (int(active[-1]) // 64 + 1) * 64))
            if bool(((selected < left) | (selected >= right)).any()):
                raise ValueError("Trimming would remove a scored token's predecessor")
            ids, attention = ids[:,left:right], attention[:,left:right]
            positions = positions[...,left:right]
            selected = selected - left
    with torch.autocast(device_type=actor.device_name, dtype=actor.param_dtype):
        with compact_head(actor.actor_module, responses.index_select(1, columns), temperature, chunk_size, calculate_entropy):
            output = actor.actor_module(input_ids=ids, attention_mask=attention,
                position_ids=positions, use_cache=False, logits_to_keep=selected)
    statistics = output.logits
    result = {"log_probs": statistics.new_zeros(responses.shape).index_copy(1, columns, statistics[..., 0]) * mask}
    if calculate_entropy:
        result["entropys"] = statistics.new_zeros(responses.shape).index_copy(1, columns, statistics[..., 1]) * mask
    return result
