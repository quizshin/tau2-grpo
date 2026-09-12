"""Numerical regression for native padded Qwen3.5 policy-only projections."""
import copy

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from test_qwen35 import tiny_model
from verl.utils.qwen35_loss_projection import response_projection
from verl.workers.actor.dp_actor import DataParallelPPOActor


def actor(model, device):
    result = object.__new__(DataParallelPPOActor)
    result.actor_module = model
    result.config = OmegaConf.create({"calculate_sum_pi_squared": False,
                                     "entropy_checkpointing": False})
    result.use_remove_padding = result.use_ulysses_sp = False
    result.use_fused_kernels = result.use_prefix_grouper = False
    result.device_name = device
    result.param_dtype = torch.bfloat16
    return result


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("empty", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_projected_native_forward_preserves_masked_values_and_lora_gradients(monkeypatch, tied, empty, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA comparison requires a remote GPU")
    torch.manual_seed(42)
    base = tiny_model(tied)
    base.model.visual.requires_grad_(False)
    full = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0,
        target_modules=["q_proj", "in_proj_qkv"], task_type="CAUSAL_LM")).to(device)
    with torch.no_grad():
        for name, parameter in full.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=.01)
    projected = copy.deepcopy(full)
    original_linear = F.linear
    head_chunks = []

    def record_linear(inputs, weight, bias=None):
        if weight is projected.get_output_embeddings().weight:
            head_chunks.append(inputs.shape[-2])
        return original_linear(inputs, weight, bias)

    monkeypatch.setattr(F, "linear", record_linear)
    monkeypatch.setenv("VERL_QWEN35_HEAD_CHUNK_SIZE", "2")
    # Different prompt padding and interior observations exercise recurrent state.
    ids = torch.tensor([[0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0],
                        [0, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0]], device=device)
    attention = ids.ne(0).long()
    positions = (attention.cumsum(-1) - 1).clamp_min(0)
    mask = torch.tensor([[1, 0, 0, 1, 1, 0, 0],
                         [0, 1, 0, 1, 0, 0, 0]], device=device)
    if empty:
        mask.zero_()
    batch = {"input_ids": ids, "responses": ids[:, 5:], "attention_mask": attention,
             "position_ids": positions, "response_mask": mask}
    seen = []
    hook = projected.get_output_embeddings().register_forward_pre_hook(
        lambda module, args: seen.append(args[0].shape[-2]))
    monkeypatch.setenv("VERL_QWEN35_LOSS_ONLY_LOGITS", "0")
    expected = actor(full, device)._forward_micro_batch(batch, .7, calculate_entropy=True)
    monkeypatch.setenv("VERL_QWEN35_LOSS_ONLY_LOGITS", "1")
    actual = actor(projected, device)._forward_micro_batch(batch, .7, calculate_entropy=True)
    hook.remove()
    assert seen == [max(1, int(mask.bool().any(0).sum()))]
    for key in ("log_probs", "entropys"):
        torch.testing.assert_close(actual[key] * mask, expected[key] * mask,
                                   atol=.04, rtol=.01)
    weights = torch.arange(1, 8, device=device).float()[None, :] * mask
    for output in (actual, expected):
        ((output["log_probs"] + .02 * output["entropys"]) * weights).sum().backward()
    assert head_chunks and max(head_chunks) <= 2
    assert "forward" not in projected.get_output_embeddings().__dict__
    for (name, p), (_, q) in zip(full.named_parameters(), projected.named_parameters()):
        if p.grad is not None:
            assert q.grad is not None, name
            torch.testing.assert_close(p.grad, q.grad, atol=.01, rtol=.04, msg=name)
    assert any(p.grad is not None for p in projected.parameters())
    if empty:
        assert all(p.grad is None or p.grad.count_nonzero() == 0 for p in projected.parameters())


def test_projection_rejects_missing_mask_and_empty_prompt():
    ids = torch.ones(1, 8, dtype=torch.long)
    responses = ids[:, 4:]
    with pytest.raises(ValueError, match="response_mask"):
        response_projection(ids, responses, None)
    with pytest.raises(ValueError, match="nonempty prompt"):
        response_projection(responses, responses, torch.ones_like(responses))
