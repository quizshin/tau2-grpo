from types import SimpleNamespace

import pytest
import torch
from verl.utils.qwen35_full_weight_audit import FullLanguageWeightAudit


def fake_model(params):
    return SimpleNamespace(named_parameters=lambda remove_duplicate=False: iter(params.items()))


def test_packed_mapping_checks_values_and_rejects_missing_slices():
    audit = FullLanguageWeightAudit()
    prefix = "model.language_model.layers.0.self_attn."
    weights = [(prefix + part + ".weight", torch.full((size, 3), float(i)))
               for i, (part, size) in enumerate((("q_proj", 4), ("k_proj", 2), ("v_proj", 2)))]
    audit.capture(weights[:2])
    target = "language_model.model.layers.0.self_attn.qkv_proj.weight"
    params = {target: torch.cat([t for _, t in weights])}
    with pytest.raises(ValueError, match="Incomplete packed"):
        audit.compare(fake_model(params))
    audit.capture(weights[2:])
    assert audit.compare(fake_model(params))["all_active_parameters_match"]
    params[target][0, 0] += 1
    assert not audit.compare(fake_model(params))["all_active_parameters_match"]


def test_tied_alias_and_explicit_vision_exclusion_do_not_hide_missing_weights():
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    audit = FullLanguageWeightAudit()
    audit.capture([("model.language_model.embed_tokens.weight", tensor),
                   ("model.visual.some.weight", torch.ones(1))])
    params = {"language_model.model.embed_tokens.weight": tensor,
              "language_model.lm_head.weight": tensor}
    report = audit.compare(fake_model(params))
    assert report["all_active_parameters_match"] and len(report["tied_aliases"]) == 1
    assert report["excluded_vision_names"] == ["model.visual.some.weight"]
    params["language_model.model.norm.weight"] = torch.ones(3)
    assert not audit.compare(fake_model(params))["all_active_parameters_match"]


def test_unknown_tensor_and_duplicate_are_errors():
    audit = FullLanguageWeightAudit()
    with pytest.raises(ValueError, match="Unsupported"):
        audit.capture([("unexpected.weight", torch.ones(1))])
    audit = FullLanguageWeightAudit()
    weights = [("model.language_model.norm.weight", torch.ones(2))]
    audit.capture(weights)
    with pytest.raises(ValueError, match="Duplicate"):
        audit.capture(weights)


@pytest.mark.parametrize("sources,target", [
    (["gate_proj", "up_proj"], "gate_up_proj"),
    (["in_proj_qkv", "in_proj_z"], "in_proj_qkvz"),
    (["in_proj_b", "in_proj_a"], "in_proj_ba"),
])
def test_all_dense_fusions_preserve_documented_order(sources, target):
    tensors = [torch.full((i + 2, 3), float(i + 1)) for i in range(len(sources))]
    audit = FullLanguageWeightAudit()
    audit.capture([(f"model.language_model.layers.0.m.{name}.weight", value)
                   for name, value in zip(sources, tensors)])
    model = fake_model({f"language_model.model.layers.0.m.{target}.weight": torch.cat(tensors)})
    assert audit.compare(model)["all_active_parameters_match"]
