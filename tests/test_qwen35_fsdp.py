"""Real FSDP regression: frozen tied embeddings and accumulated LoRA updates."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from test_qwen35 import tiny_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from transformers import AutoModelForImageTextToText
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, resolve_fsdp_use_orig_params


@pytest.mark.parametrize("tied", [False, True])
def test_lora_fsdp_accumulates_multiple_microbatches_with_tied_embeddings(tmp_path, tied):
    torch.manual_seed(42)
    base = tiny_model(tied).to(torch.bfloat16)
    base.save_pretrained(tmp_path / "base", save_original_format=False)
    base = AutoModelForImageTextToText.from_pretrained(tmp_path / "base", dtype=torch.bfloat16)
    assert resolve_fsdp_use_orig_params(base, True, False) is True
    base.enable_input_require_grads()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(
        base,
        LoraConfig(
            r=4, lora_alpha=8, target_modules=["q_proj", "in_proj_qkv"], task_type="CAUSAL_LM"
        ),
    )
    use_orig = resolve_fsdp_use_orig_params(model, True, True)
    assert use_orig is (not tied)
    frozen_before = base.get_input_embeddings().weight.detach().clone()
    dist.init_process_group(
        "gloo", init_method="file://" + str(tmp_path / "rendezvous"), rank=0, world_size=1
    )
    try:
        wrapped = FSDP(
            model,
            device_id=torch.device("cpu"),
            auto_wrap_policy=get_fsdp_wrap_policy(model, is_lora=True),
            use_orig_params=use_orig,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
            ),
        )
        optimizer = torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad], lr=1e-3)
        trainable_before = [p.detach().clone() for p in wrapped.parameters() if p.requires_grad]
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        for _ in range(2):
            wrapped.eval()
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                wrapped(input_ids=tokens, use_cache=False)
                with wrapped.disable_adapter():
                    wrapped(input_ids=tokens, use_cache=False)
            wrapped.train()
            optimizer.zero_grad()
            for _ in range(2):
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    loss = (
                        wrapped(input_ids=tokens, use_cache=False).logits.float().square().mean()
                        / 2
                    )
                loss.backward()
            norm = wrapped.clip_grad_norm_(1.0)
            assert torch.isfinite(norm) and norm > 0
            optimizer.step()
        trainable_after = [p.detach() for p in wrapped.parameters() if p.requires_grad]
        assert any(not torch.equal(a, b) for a, b in zip(trainable_before, trainable_after))
        with FSDP.summon_full_params(wrapped):
            assert torch.equal(base.get_input_embeddings().weight, frozen_before)
            if tied:
                assert base.get_input_embeddings().weight is base.get_output_embeddings().weight
    finally:
        dist.destroy_process_group()


def test_rollout_weight_audit_checks_actual_buffers_and_rejects_wrong_base(tmp_path):
    import json

    from safetensors.torch import save_file
    from verl.utils.qwen35_weight_audit import audit_qwen35_rollout_weights

    model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.linear_attn = torch.nn.Module()
    layer.linear_attn.conv1d = torch.nn.Module()
    layer.linear_attn.conv1d.base_layer = torch.nn.Linear(4, 8, bias=False)
    layer.linear_attn.register_buffer("lora_b_stacked", torch.zeros(1, 8, 2))
    model.layers = torch.nn.ModuleList([layer])
    base = tmp_path / "base"
    base.mkdir()
    save_file(
        {
            "model.language_model.layers.0.linear_attn.conv1d.weight": layer.linear_attn.conv1d.base_layer.weight.detach().clone()
        },
        base / "model.safetensors",
    )
    first = json.loads(
        Path(audit_qwen35_rollout_weights(model, base, tmp_path / "audit", "base")).read_text()
    )
    assert first["all_conv_match"]
    assert first["lora_gpu"][0]["nonzero"] == 0
    layer.linear_attn.lora_b_stacked.fill_(1)
    second = json.loads(
        Path(audit_qwen35_rollout_weights(model, base, tmp_path / "audit", "adapter")).read_text()
    )
    assert second["lora_gpu"][0]["sha256"] != first["lora_gpu"][0]["sha256"]
    with torch.no_grad():
        layer.linear_attn.conv1d.base_layer.weight.add_(1)
    with pytest.raises(RuntimeError, match="differ from the base checkpoint"):
        audit_qwen35_rollout_weights(model, base, tmp_path / "audit", "base")
    native = "model.language_model.layers.0.linear_attn.conv1d.weight"
    incoming = {native: layer.linear_attn.conv1d.base_layer.weight.detach().clone()}
    full = json.loads(Path(audit_qwen35_rollout_weights(
        model, base, tmp_path / "audit", "full", expected_conv=incoming
    )).read_text())
    assert full["all_conv_match"]
    assert full["comparison_source"] == "incoming actor weights"
    with torch.no_grad():
        layer.linear_attn.conv1d.base_layer.weight.add_(1)
    with pytest.raises(RuntimeError, match="incoming actor weights"):
        audit_qwen35_rollout_weights(model, base, tmp_path / "audit", "full", expected_conv=incoming)
    with pytest.raises(RuntimeError, match="incoming actor weights"):
        audit_qwen35_rollout_weights(model, base, tmp_path / "audit", "full", expected_conv={})
