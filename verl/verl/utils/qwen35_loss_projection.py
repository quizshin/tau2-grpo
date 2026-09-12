# Tau3-GRPO local patch: preserve native padded attention/recurrent state while
# omitting vocabulary projections that cannot contribute to the policy loss.
from contextlib import contextmanager

import torch
from torch.utils.checkpoint import checkpoint

from verl.utils.torch_functional import logprobs_from_logits


def response_projection(input_ids, responses, response_mask):
    if response_mask is None or response_mask.shape != responses.shape:
        raise ValueError("Qwen3.5 loss projection requires the complete response_mask")
    prompt_length = input_ids.shape[-1] - responses.shape[-1]
    if prompt_length < 1 or responses.shape[-1] < 1:
        raise ValueError("Qwen3.5 loss projection requires a nonempty prompt and response")
    columns = response_mask.bool().any(dim=0).nonzero(as_tuple=True)[0]
    if columns.numel() == 0:
        # DF can zero a whole local micro-batch. Keep a connected forward/backward
        # on every FSDP rank; the existing response mask makes its loss zero.
        columns = torch.zeros(1, dtype=torch.long, device=input_ids.device)
    return columns, columns + prompt_length - 1


def restore_response_columns(values, columns, response_length):
    """Unused columns are zero; participating columns retain their autograd path."""
    return values.new_zeros((values.shape[0], response_length)).index_copy(1, columns, values)


@contextmanager
def compact_lm_head(model, labels, temperature, chunk_size, entropy_fn=None, sum_pi_squared_fn=None):
    """Compute compact token statistics inside the native/FSDP model forward.

    Only the frozen linear head is temporarily intercepted. The transformer,
    padding and recurrent state are unchanged. Checkpoint each head projection
    together with its reductions so backward never retains a full vocabulary
    tensor for every policy token. The original bound forward is captured for
    recomputation after this context restores the module method.
    """
    head = model.get_output_embeddings()
    if not isinstance(head, torch.nn.Linear) or head.weight.requires_grad:
        raise ValueError("Chunked Qwen3.5 statistics require a frozen native linear lm_head")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    original = head.forward
    had_instance_forward = "forward" in head.__dict__

    def statistics(hidden, targets):
        logits = original(hidden)
        logits.div_(temperature)
        values = [logprobs_from_logits(logits, targets)]
        if entropy_fn is not None:
            values.append(entropy_fn(logits))
        if sum_pi_squared_fn is not None:
            values.append(sum_pi_squared_fn(logits))
        return torch.stack(values, dim=-1)

    def forward(hidden):
        if hidden.shape[:2] != labels.shape:
            raise ValueError("Projected hidden states and policy labels are misaligned")
        chunks = []
        for start in range(0, hidden.shape[1], chunk_size):
            part = hidden[:, start:start + chunk_size]
            targets = labels[:, start:start + chunk_size]
            if torch.is_grad_enabled() and part.requires_grad:
                # The frozen linear head and reductions use no randomness.
                value = checkpoint(statistics, part, targets, use_reentrant=False,
                                   preserve_rng_state=False)
            else:
                value = statistics(part, targets)
            chunks.append(value)
        return torch.cat(chunks, dim=1)

    head.forward = forward
    try:
        yield
    finally:
        if had_instance_forward:
            head.forward = original
        else:
            del head.forward
