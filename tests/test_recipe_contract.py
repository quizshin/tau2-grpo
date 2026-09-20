import json
from pathlib import Path

import pytest

from tau3_grpo.evaluation.rewards import recipe

LEGACY = Path(__file__).parent / "fixtures/irc_legacy_v1_synthetic.json"


def test_known_legacy_artifact_keeps_original_identity():
    raw = LEGACY.read_text()
    artifact = recipe.load_frozen_recipe(LEGACY, manifest_sha256="fixture")
    assert artifact == json.loads(raw)
    assert artifact["schema"] == "mt_gtpo_irc_recipe_v1"
    assert LEGACY.read_text() == raw


@pytest.mark.parametrize("source", ["analysis/calibrate_paper_rewards.py", "algorithms/mt_gtpo.py"])
def test_unknown_legacy_implementation_is_not_blessed(tmp_path, source):
    artifact = json.loads(LEGACY.read_text())
    artifact["implementation_sha256"][source] = "unknown"
    artifact["sha256"] = recipe.digest({k: v for k, v in artifact.items() if k != "sha256"})
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="implementation changed"):
        recipe.load_frozen_recipe(path)


def test_legacy_recipe_rejects_numerical_source_drift(monkeypatch):
    hashes = recipe.implementation_hashes()
    hashes["evaluation/process_reward.py"] = "changed"
    monkeypatch.setattr(recipe, "implementation_hashes", lambda: hashes)
    with pytest.raises(ValueError, match="implementation changed"):
        recipe.load_frozen_recipe(LEGACY)
