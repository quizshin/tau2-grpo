"""Publication boundaries must preserve the supported two-card entry points."""

import pytest
import yaml

from tau3_grpo.configuration import load_config_with_sources
from tau3_grpo.paths import CODE_ROOT


@pytest.mark.parametrize("name", ["grpo", "arpo", "mt_gtpo"])
def test_catalog_profiles_resolve_to_two_card_shared_layout(name):
    catalog = yaml.safe_load((CODE_ROOT / "configs/experiments/catalog.yaml").read_text())
    profile = CODE_ROOT / catalog["rl"][name]["profile"]
    config, _ = load_config_with_sources(profile)
    env = config["launch"]["environment"]
    assert str(env["POLICY_GPUS"]) == "2"
    assert str(env["TAU3_POLICY_CUDA_DEVICES"]) == "0,1"
    assert str(env["TAU3_USER_CUDA_DEVICES"]) == "1"
    assert str(env["TAU3_SIMULATOR_COLOCATED_SLEEP"]) == "1"
    assert any(
        "SleepingSimulatorAgentLoopManager" in item for item in config["launch"]["overrides"]
    )
    if name == "mt_gtpo":
        assert config["launch"]["terminal_reward_protocol"] == "tau2_native_v1"


def test_all_config_include_chains_exist():
    for path in (CODE_ROOT / "configs").rglob("*.yaml"):
        load_config_with_sources(path)


def test_frozen_reference_manifests_are_source_inputs():
    from tau3_grpo.data.manifest import read_manifest

    path = CODE_ROOT / "data/manifests/rl_curriculum50_20260912/areal_airline_train_seed42.jsonl"
    rows = read_manifest(path)
    assert len(rows) == 50
    assert len({row.task_id for row in rows}) == 50
    assert not (CODE_ROOT / "results/analysis/rl_curriculum50_20260912/manifests").exists()


def test_cli_requires_explicit_hardware_before_loading_runtime():
    # Inspect argparse declarations without importing optional training dependencies.
    import ast

    tree = ast.parse((CODE_ROOT / "tau3_grpo/training/rl/runner.py").read_text())
    options = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--profile"
    ]
    assert len(options) == 1
    assert any(
        kw.arg == "required" and isinstance(kw.value, ast.Constant) and kw.value.value is True
        for kw in options[0].keywords
    )


def test_publication_has_no_retired_hardware_install_tree():
    assert not (CODE_ROOT / "env_info/paratera").exists()
    assert not (CODE_ROOT / "env_info/historical").exists()
    assert not (CODE_ROOT / "requirements-local.txt").exists()
    profiles = list((CODE_ROOT / "configs").rglob("*.yaml"))
    assert not [str(path) for path in profiles if "5090" in path.name or "paratera" in path.name]
