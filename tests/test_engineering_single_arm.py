"""Reject broader or repeated GPU attempts before any service startup."""
import importlib.util
import json

import pytest
import yaml

from tau3_grpo.paths import CODE_ROOT


@pytest.fixture
def controller(monkeypatch):
    folder = CODE_ROOT / "env_info/a800_20260919"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("engineering_updates", folder / "architecture_updates.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_plan(root):
    plan = {"b-tau_gigpo-df0": {"estimator": "tau_gigpo", "filtering": False, "command": []}}
    (root / "phase-b-plan.json").write_text(json.dumps(plan))
    config = {"algorithm": {"adv_estimator": "tau_gigpo", "dynamic_filter": {"enable": False}},
              "trainer": {"total_training_steps": 2, "save_freq": 1, "test_freq": -1},
              "data": {"train_batch_size": 8}, "actor_rollout_ref": {"rollout": {"n": 8}}}
    path = root / "b-tau_gigpo-df0/resolved-hydra.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(config))
    return plan, config, path


def test_single_arm_plan_accepts_exact_approved_scope(controller, tmp_path):
    make_plan(tmp_path)
    label, plan = controller.single_arm_plan(tmp_path)
    assert label == "b-tau_gigpo-df0"
    assert plan["estimator"] == "tau_gigpo"


@pytest.mark.parametrize("key,value", [("total_training_steps", 3), ("save_freq", 10),
                                      ("test_freq", 1)])
def test_protocol_drift_is_rejected(controller, tmp_path, key, value):
    _, config, path = make_plan(tmp_path)
    config["trainer"][key] = value
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="protocol mismatch"):
        controller.single_arm_plan(tmp_path)


def test_extra_algorithm_is_rejected(controller, tmp_path):
    plan, _, _ = make_plan(tmp_path)
    plan["b-mt_gtpo-df0"] = {"estimator": "mt_gtpo", "filtering": False}
    (tmp_path / "phase-b-plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="only GiGPO"):
        controller.single_arm_plan(tmp_path)


def test_existing_attempt_never_resets_budget_or_starts_gpu(controller, tmp_path, monkeypatch):
    make_plan(tmp_path)
    (tmp_path / "authorization.json").write_text(json.dumps({"wall_seconds": 5400,
                                                            "max_training_candidates": 128}))
    budget = tmp_path / "budget.json"
    budget.write_text('{"deadline_unix": 1}')
    monkeypatch.setattr(controller.subprocess, "check_output",
                        lambda *args, **kwargs: pytest.fail("GPU preflight should not be reached"))
    with pytest.raises(ValueError, match="no automatic retry"):
        controller.execute_single(tmp_path)
    assert budget.read_text() == '{"deadline_unix": 1}'


def test_timeout_is_explicit(controller):
    with pytest.raises(TimeoutError, match="90-minute"):
        controller._raise_budget_timeout()


def test_accepted_step_boundary_is_not_rechecked_during_next_update(controller):
    checked = set()
    rows = [{"step": 1}]
    assert not controller.check_boundary_once(rows, checked, 2400)
    assert not controller.check_boundary_once(rows, checked, 1000)
    assert checked == {1}


def test_insufficient_budget_at_first_boundary_stops(controller):
    checked = set()
    assert controller.check_boundary_once([{"step": 1}], checked, 1000)
    assert not controller.check_boundary_once([{"step": 2}], checked, 500)
