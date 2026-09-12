"""Full-parameter PPO output/gradient checks, including trainable tied heads."""
import copy

import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

from test_qwen35 import tiny_model
from verl.workers.actor.dp_actor import DataParallelPPOActor


def actor(model, device="cuda"):
    a = object.__new__(DataParallelPPOActor)
    a.actor_module = model
    a.config = OmegaConf.create({"calculate_sum_pi_squared": False, "entropy_checkpointing": False})
    a.use_remove_padding = a.use_ulysses_sp = a.use_fused_kernels = a.use_prefix_grouper = False
    a.device_name, a.param_dtype = device, torch.bfloat16
    return a


def batch(device="cuda", empty=False):
    ids = torch.tensor([[0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0],
                        [0, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0]], device=device)
    attn = ids.ne(0).long()
    mask = torch.tensor([[1, 0, 0, 1, 1, 0, 0], [0, 1, 0, 1, 0, 0, 0]], device=device)
    if empty:
        mask.zero_()
    return {"input_ids": ids, "responses": ids[:, 5:], "attention_mask": attn,
        "position_ids": (attn.cumsum(-1)-1).clamp_min(0), "response_mask": mask}


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("empty", [True, False])
def test_full_parameter_outputs_and_gradients(monkeypatch, tied, empty):
    if not torch.cuda.is_available():
        pytest.skip("GPU numerical validation")
    torch.manual_seed(42)
    base = tiny_model(tied).cuda()
    base.model.visual.requires_grad_(False)
    models = [base, copy.deepcopy(base)]
    data = batch(empty=empty)
    results = []
    for mode, model in enumerate(models):
        monkeypatch.setenv("VERL_QWEN35_COMPACT_HEAD", str(mode))
        monkeypatch.setenv("VERL_QWEN35_COMPACT_CHUNK", "2")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        out = actor(model)._forward_micro_batch(data, .7, calculate_entropy=True)
        weights = torch.arange(1, 8, device="cuda").float()[None, :] * data["response_mask"]
        ((out["log_probs"] + .02*out["entropys"]) * weights).sum().backward()
        results.append(out)
    mask = data["response_mask"]
    for key in ("log_probs", "entropys"):
        torch.testing.assert_close(results[0][key]*mask, results[1][key]*mask, atol=.04, rtol=.01)
    for (name, p), (_, q) in zip(models[0].named_parameters(), models[1].named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=.01, rtol=.04)
    head = models[1].get_output_embeddings().weight
    assert head.requires_grad and head.grad is not None
    assert "forward" not in models[1].get_output_embeddings().__dict__
    if empty:
        assert all(p.grad is None or not p.grad.count_nonzero() for p in models[1].parameters())
    else:
        assert head.grad.count_nonzero()


@pytest.mark.parametrize("tied", [True, False])
def test_right_padding_trim_preserves_singleton_full_gradients(monkeypatch, tied):
    if not torch.cuda.is_available():
        pytest.skip("GPU numerical validation")
    torch.manual_seed(42)
    base = tiny_model(tied).cuda()
    with torch.no_grad():
        base.get_input_embeddings().weight[0].normal_(std=.1)
    base.model.visual.requires_grad_(False)
    models = [base, copy.deepcopy(base)]
    data = {k:v[:1] for k,v in batch().items()}
    # Cross multiple native GDN chunk boundaries, with real nonzero left-pad
    # embeddings and a singleton batch, matching production microbatch size.
    for key in data:
        data[key] = torch.nn.functional.pad(data[key], (0, 128))
    results = []
    for mode, model in enumerate(models):
        monkeypatch.setenv("VERL_QWEN35_COMPACT_HEAD", str(mode))
        monkeypatch.setenv("VERL_QWEN35_TRIM_PADDING", "right")
        monkeypatch.setenv("VERL_QWEN35_COMPACT_CHUNK", "2")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        out = actor(model)._forward_micro_batch(data, 1., calculate_entropy=True)
        ((out["log_probs"] + .02*out["entropys"]) * data["response_mask"]).sum().backward()
        results.append(out)
    for key in ("log_probs", "entropys"):
        torch.testing.assert_close(results[0][key]*data["response_mask"],
            results[1][key]*data["response_mask"], atol=.04, rtol=.01)
    for (name,p),(_,q) in zip(models[0].named_parameters(), models[1].named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,atol=.01,rtol=.04)


def test_compute_log_prob_keeps_response_mask(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("GPU numerical validation")
    monkeypatch.setenv("VERL_QWEN35_COMPACT_HEAD", "1")
    monkeypatch.setenv("VERL_QWEN35_COMPACT_CHUNK", "2")
    torch.manual_seed(42)
    a = actor(tiny_model(False).cuda())
    data = batch()
    packet = DataProto.from_dict(tensors=data, meta_info={
        "micro_batch_size": 1, "temperature": 1., "use_dynamic_bsz": False})
    monkeypatch.setenv("VERL_QWEN35_COMPACT_HEAD", "0")
    expected = a.compute_log_prob(packet, calculate_entropy=True)
    monkeypatch.setenv("VERL_QWEN35_COMPACT_HEAD", "1")
    scored = a.compute_log_prob(packet, calculate_entropy=True)
    for key in ("log_probs", "entropys"):
        torch.testing.assert_close(scored[key], expected[key]*data["response_mask"], atol=.01, rtol=.01)
        assert torch.count_nonzero(scored[key][~data["response_mask"].bool()]) == 0
