import importlib.util
import json
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from tau3_grpo.algorithms.mt_gtpo import compute_mt_gtpo
from tau3_grpo.analysis.calibrate_paper_rewards import (
    DEFAULT_IRC,
    alignment_issues,
    calibrate_round,
    digest,
    freeze_recipe,
    load_frozen_recipe,
    load_round,
    main,
    rescore,
    summarize,
    task_split,
    validate_irc,
)
from tau3_grpo.evaluation.process_reward import DEFAULT_WEIGHTS, reward_settings, score_turns
from tau3_grpo.paths import CODE_ROOT

ALGORITHM = {"gamma": .9, "lambda_outcome": .3, "eps": 1e-6, "min_group_size": 2}
PAPER = reward_settings({"mode": "paper", "version": "paper_v1"})


def simple_config(version="paper_v1"):
    c = deepcopy(DEFAULT_IRC)
    c.update(min_support=2, intended_signs={"gold_exact": 1, "error": -1},
             fixed_weights={k: 0.0 for k in DEFAULT_WEIGHTS if k not in {"gold_exact", "error"}})
    if version in {"paper_env_split_v3", "paper_env_split_v4"}:
        c["reward_version"] = version
        c["intended_signs"] = {"gold_write": 1, "error": -1}
        c["fixed_weights"]["gold_exact"] = 0.0
        c["fixed_weights"]["gold_read"] = 0.0
    if version == "paper_env_split_v4":
        c["fixed_weights"]["generic"] = 0.0
    return c


def make_update(task_ids=("a", "b", "c", "d"), *, all_fail=False, harmful_soft=False):
    processes, outcomes, tasks, uids = [], [], [], []
    gold = {"name": "cancel_reservation", "arguments": {"reservation_id": "A"}}
    if harmful_soft:
        gold["arguments"]["other"] = 1
    for task in task_ids:
        for success in (True, False, True, False):
            outcome = float(success and not all_fail)
            args = gold["arguments"] if success else {"reservation_id": "B"}
            if harmful_soft and not success:
                args = {"reservation_id": "A", "other": 2}
            turns = [{"schema": "tau3_turn_v1", "turn_index": 0, "token_span": [0, 3],
                      "tool_calls": [{"name": gold["name"], "arguments": args,
                                      "error": not success and not harmful_soft}]}]
            processes.append(score_turns(turns, [gold], ["DB"], PAPER, official_outcome=outcome))
            outcomes.append(outcome)
            tasks.append(task)
            uids.append(task)
    return {"processes": processes, "outcomes": outcomes, "task_ids": tasks,
            "uids": uids, "algorithm": ALGORITHM}


def saved_rows(update):
    p = update["processes"]
    mask = np.ones((len(p), 3))
    _, _, details = compute_mt_gtpo(update["outcomes"], update["uids"],
                                    [x["turn_rewards"] for x in p],
                                    [x["turn_spans"] for x in p], mask, **ALGORITHM)
    rows = []
    for i, process in enumerate(p):
        replay = {"schema": "mt_gtpo_replay_v1", "uid": update["uids"][i],
                  "sampled_group_size": 4, "outcome": update["outcomes"][i],
                  "settings": ALGORITHM, "response_mask": mask[i].tolist(), "process": process,
                  "filtered_response_mask": mask[i].tolist(), "dynamic_filter": {"enable": False},
                  "episode_advantage": details["episode_advantages"][i],
                  "turn_returns": details["turn_returns"][i],
                  "turn_advantages": details["turn_advantages"][i]}
        rows.append({"task_id": update["task_ids"][i], "mt_gtpo_replay_json": json.dumps(replay)})
    return rows


def report_for(update=None, recipe=PAPER):
    update = make_update() if update is None else update
    c = simple_config(recipe["version"])
    result = calibrate_round([update], recipe, {"a", "b"}, {"c", "d"}, c)
    result["sources"] = [{"sha256": "fixture", "rows": 16}]
    return {"rounds": [result], "algorithm": ALGORITHM, "irc": c,
            "train_manifest": {"sha256": "fixture"},
            "calibration_tasks": ["a", "b"], "holdout_tasks": ["c", "d"]}


def test_passing_round_freeze_and_exact_recipe_loading(tmp_path):
    report = report_for()
    assert report["rounds"][0]["passed"]
    artifact = freeze_recipe(report)
    assert artifact["reward"]["paper_options"]["soft_scoring"] == "constant"
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(artifact))
    assert load_frozen_recipe(path, manifest_sha256="fixture") == artifact
    with pytest.raises(ValueError, match="manifest"):
        load_frozen_recipe(path, manifest_sha256="different")
    artifact["reward"]["weights"]["gold_exact"] = 5
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="hash"):
        load_frozen_recipe(path)


def test_no_variance_insufficient_support_and_unknown_are_not_convergence():
    report = report_for(make_update(all_fail=True))
    assert not report["rounds"][0]["passed"]
    with pytest.raises(ValueError, match="cannot freeze"):
        freeze_recipe(report)
    obs = rescore([make_update()], PAPER)
    s = summarize(obs, simple_config())
    s["tiers"]["unknown"]["trajectory_presence"] = 1
    assert "unknown_tool_tier_present" in alignment_issues(s, simple_config())
    s["tiers"]["error"]["mean_hybrid_advantage"] = .2
    assert "error: mean_hybrid_advantage_misaligned" in alignment_issues(s, simple_config())


def test_negative_soft_correlation_is_flagged_not_relabeled_as_success():
    c = simple_config()
    c["intended_signs"] = {"gold_exact": 1, "soft_match": 1}
    c["fixed_weights"] = {k: 0 for k in DEFAULT_WEIGHTS if k not in c["intended_signs"]}
    result = calibrate_round([make_update(harmful_soft=True)], PAPER, {"a", "b"}, {"c", "d"}, c)
    assert result["candidate_recipe"]["weights"]["soft_match"] < 0
    assert not result["passed"]
    assert "soft_match: proposed_weight_conflicts_with_intended_sign" in result["issues"]["proposal"]


def test_holdout_outcomes_do_not_change_fitted_weights():
    update = make_update()
    original = calibrate_round([update], PAPER, {"a", "b"}, {"c", "d"}, simple_config())
    altered = make_update()
    for i, task in enumerate(altered["task_ids"]):
        if task in {"c", "d"}:
            altered["outcomes"][i] = 1 - altered["outcomes"][i]
    changed = calibrate_round([altered], PAPER, {"a", "b"}, {"c", "d"}, simple_config())
    assert original["candidate_recipe"] == changed["candidate_recipe"]
    assert not changed["passed"]


def test_train_only_groups_replay_and_duplicate_round_rejection(tmp_path):
    path = tmp_path / "update.jsonl"
    rows = saved_rows(make_update())
    path.write_text("\n".join(json.dumps(row) for row in rows))
    seen = set()
    updates, _ = load_round([path], {"a", "b", "c", "d"}, seen)
    assert len(updates[0]["outcomes"]) == 16
    with pytest.raises(ValueError, match="duplicate buffer"):
        load_round([path], {"a", "b", "c", "d"}, seen)
    with pytest.raises(ValueError, match="outside"):
        load_round([path], {"a", "b"}, set())
    path.write_text("\n".join(json.dumps(row) for row in rows[1:]))
    with pytest.raises(ValueError, match="incomplete"):
        load_round([path], {"a", "b", "c", "d"}, set())


@pytest.mark.parametrize("version", ["paper_v1", "paper_env_v2", "paper_env_split_v3", "paper_env_split_v4"])
def test_cli_reports_failure_without_freezing_and_success_with_provenance(tmp_path, version):
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("\n".join(json.dumps({"task_id": t, "split": "train"}) for t in "abcd"))
    cfg = tmp_path / "config.json"
    c = simple_config(version)
    c["holdout_fraction"] = .5
    cfg.write_text(json.dumps(c))
    buf = tmp_path / "batch.jsonl"
    buf.write_text("\n".join(json.dumps(r) for r in saved_rows(make_update())))
    args = ["--round", str(buf), "--train-manifest", str(manifest), "--config", str(cfg),
            "--reward-version", version]
    assert main([*args, "--output-dir", str(tmp_path / "pass")]) == 0
    artifact = load_frozen_recipe(tmp_path / "pass/frozen-recipe.json")
    assert artifact["reward"]["version"] == version
    assert set(artifact["calibration_tasks"]).isdisjoint(artifact["holdout_tasks"])
    buf.write_text("\n".join(json.dumps(r) for r in saved_rows(make_update(all_fail=True))))
    assert main([*args, "--output-dir", str(tmp_path / "fail")]) == 2
    assert not (tmp_path / "fail/frozen-recipe.json").exists()
    assert (tmp_path / "fail/report.json").exists()
    manifest.write_text(json.dumps({"task_id": "a", "split": "selection"}))
    with pytest.raises(ValueError, match="training-only"):
        main([*args, "--output-dir", str(tmp_path / "invalid")])


def test_split_is_task_stable_and_thresholds_validated():
    assert task_split(list("abcdef"), fraction=.3, seed=42) == task_split(list("fedcba"), fraction=.3, seed=42)
    for change in [{"alpha": float("nan")}, {"min_support": 0}, {"eta": 1}, {"delta": -1}]:
        with pytest.raises(ValueError):
            validate_irc({**DEFAULT_IRC, **change})


def test_fixed_weights_can_define_nonzero_reference_anchor():
    config = deepcopy(DEFAULT_IRC)
    config["intended_signs"] = {"state_change": -1, "error": -1}
    config["fixed_weights"] = {
        "gold_exact": 1.0, "soft_match": 0.0, "read_only": 0.0,
        "duplicate": 0.0, "message": 0.0, "unknown": 0.0,
    }
    with pytest.raises(ValueError, match="explicit fixed anchor"):
        validate_irc(config)
    config["fixed_anchor_policy"] = "gold_reference_v1"
    validate_irc(config)
    for tier, value in [("gold_exact", 2), ("unknown", 1), ("error", float("nan"))]:
        invalid = deepcopy(config)
        if tier in invalid["intended_signs"]:
            del invalid["intended_signs"][tier]
        invalid["fixed_weights"][tier] = value
        with pytest.raises(ValueError):
            validate_irc(invalid)


def anchor_config():
    c = simple_config()
    del c["intended_signs"]["gold_exact"]
    c["fixed_weights"]["gold_exact"] = 1.0
    c["fixed_anchor_policy"] = "gold_reference_v1"
    return c


def test_anchor_checks_keep_direction_and_fitted_support_requirements():
    c = anchor_config()
    summary = summarize(rescore([make_update()], PAPER), c)
    gold = summary["tiers"]["gold_exact"]
    gold.update(supported=False, support=0, rho=None)
    assert not alignment_issues(summary, c)  # anchor is prescribed, not estimated
    gold["mean_hybrid_advantage"] = -.1
    assert "gold_exact: mean_hybrid_advantage_misaligned" in alignment_issues(summary, c)
    gold["trajectory_presence"] = 0
    assert "gold_exact: insufficient_anchor_occurrences" in alignment_issues(summary, c)
    summary["tiers"]["error"]["supported"] = False
    assert "error: insufficient_support" in alignment_issues(summary, c)


def test_anchor_fitting_and_freeze_loading_enforce_fixed_weight(tmp_path):
    report = report_for()
    report["irc"] = anchor_config()
    result = calibrate_round([make_update()], PAPER, {"a", "b"}, {"c", "d"}, report["irc"])
    result["sources"] = report["rounds"][0]["sources"]
    report["rounds"] = [result]
    assert result["passed"]
    assert result["candidate_recipe"]["weights"]["gold_exact"] == 1
    artifact = freeze_recipe(report)
    path = tmp_path / "anchor.json"
    path.write_text(json.dumps(artifact))
    assert load_frozen_recipe(path) == artifact
    artifact["reward"]["weights"]["gold_exact"] = .5
    artifact["sha256"] = digest({k: v for k, v in artifact.items() if k != "sha256"})
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="fixed weights"):
        load_frozen_recipe(path)


def test_development_report_cannot_be_frozen(tmp_path):
    report = report_for()
    report["development_only"] = True
    with pytest.raises(ValueError, match="development-only"):
        freeze_recipe(report)
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("\n".join(json.dumps({"task_id": t, "split": "train"}) for t in "abcd"))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({**simple_config(), "holdout_fraction": .5}))
    buf = tmp_path / "buffer.jsonl"
    buf.write_text("\n".join(json.dumps(r) for r in saved_rows(make_update())))
    output = tmp_path / "dev"
    assert main(["--round", str(buf), "--train-manifest", str(manifest), "--config", str(config),
                 "--output-dir", str(output), "--development-only"]) == 0
    result = json.loads((output / "report.json").read_text())
    assert result["status"] == "development_checks_passed"
    assert result["development_only"]
    assert result["implementation_sha256"]
    assert not (output / "frozen-recipe.json").exists()


def controller():
    spec = importlib.util.spec_from_file_location("mtgtpo_controller", CODE_ROOT / "scripts/train/rl/run_mt_gtpo_formal.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("version", ["paper_v1", "paper_env_v2", "paper_env_split_v3", "paper_env_split_v4"])
def test_formal_controller_requires_calibration_and_accepts_frozen_recipe(tmp_path, monkeypatch, version):
    m = controller()
    with pytest.raises(ValueError, match="passed --reward-recipe"):
        m.resolve(tmp_path / "run", reward_version=version)
    monkeypatch.setenv("TAU3_RUN_ROOT", str(tmp_path))
    monkeypatch.setenv("TAU3_ROOT", str(tmp_path))
    monkeypatch.setenv("TAU3_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("TAU3_ENV_FILE", str(tmp_path / "absent"))
    _, _, snapshot = m.resolve(tmp_path / "dry", reward_version=version, allow_uncalibrated=True)
    assert snapshot["uncalibrated_initializer"]
    report = report_for(recipe=reward_settings({"mode": "paper", "version": version}))
    report["train_manifest"]["sha256"] = m.MANIFEST_SHA
    artifact = freeze_recipe(report)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(artifact))
    command, _, snapshot = m.resolve(tmp_path / "formal", reward_version=version, reward_recipe=path)
    error_arg = next(x for x in command if x.startswith("++algorithm.process_reward.weights.error="))
    assert float(error_arg.split("=", 1)[1]) == pytest.approx(-1.0)
    assert not snapshot["uncalibrated_initializer"]
    assert snapshot["irc_recipe"]["sha256"] == artifact["sha256"]
    with pytest.raises(ValueError, match="paper_v1"):
        m.resolve(tmp_path / "old", reward_version="v3", reward_recipe=path)
    other = "paper_env_v2" if version == "paper_v1" else "paper_v1"
    with pytest.raises(ValueError, match="version differs"):
        m.resolve(tmp_path / "wrong", reward_version=other, reward_recipe=path)


def test_frozen_recipe_rejects_changed_checks_even_with_new_hash(tmp_path):
    artifact = freeze_recipe(report_for())
    artifact["checks"]["holdout"]["tiers"]["error"]["mean_hybrid_advantage"] = 1
    artifact["sha256"] = digest({k: v for k, v in artifact.items() if k != "sha256"})
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="failed IRC checks"):
        load_frozen_recipe(path)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("version", ["paper_v1", "paper_env_v2", "paper_env_split_v3", "paper_env_split_v4"])
def test_paper_profile_shell_hydra_and_frozen_weights(enabled, version, tmp_path, monkeypatch):
    from hydra import compose, initialize_config_dir

    m = controller()
    for key, value in {"TAU3_ROOT": tmp_path, "TAU3_RUN_ROOT": tmp_path,
                       "TAU3_MODEL_ROOT": tmp_path / "models",
                       "TAU3_ENV_FILE": tmp_path / "absent"}.items():
        monkeypatch.setenv(key, str(value))
    report = report_for(recipe=reward_settings({"mode": "paper", "version": version}))
    report["train_manifest"]["sha256"] = m.MANIFEST_SHA
    artifact = freeze_recipe(report)
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(artifact))
    cmd, env, _ = m.resolve(tmp_path / "run", reward_version=version, reward_recipe=path,
                            dynamic_filter=enabled)
    env.update(TAU3_DRY_RUN="1", PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    output = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True, timeout=20)
    rendered = shlex.split(output.stdout.splitlines()[-1])
    overrides = [s for s in rendered if s.startswith(("algorithm.", "+algorithm.", "++algorithm."))]
    with initialize_config_dir(config_dir=str(CODE_ROOT / "verl/verl/trainer/config"), version_base=None):
        resolved = compose(config_name="ppo_trainer", overrides=overrides)
    assert reward_settings(resolved.algorithm.process_reward) == artifact["reward"]
    assert resolved.algorithm.mt_gtpo.gamma == artifact["algorithm"]["gamma"]
    assert resolved.algorithm.dynamic_filter.enable is enabled
    assert "actor_rollout_ref.model.lora_rank=0" in rendered
    assert "trainer.save_freq=10" in rendered
    assert "trainer.test_freq=10" in rendered


@pytest.mark.parametrize("version", ["paper_v1", "paper_env_v2", "paper_env_split_v3", "paper_env_split_v4"])
def test_resume_preserves_recipe_and_df(tmp_path, monkeypatch, version):
    m = controller()
    for key in ("TAU3_ROOT", "TAU3_MODEL_ROOT", "TAU3_RUN_ROOT"):
        monkeypatch.setenv(key, str(tmp_path))
    report = report_for(recipe=reward_settings({"mode": "paper", "version": version}))
    report["train_manifest"]["sha256"] = m.MANIFEST_SHA
    frozen = freeze_recipe(report)
    recipe = tmp_path / "frozen-recipe.json"
    recipe.write_text(json.dumps(frozen))
    (tmp_path / "resolved-hydra.yaml").write_text(yaml.safe_dump({"algorithm": {
        "adv_estimator": "mt_gtpo",
        "process_reward": frozen["reward"], "mt_gtpo": frozen["algorithm"],
        "dynamic_filter": {"enable": False}}}))
    checkpoint = tmp_path / "global_step_10"
    checkpoint.mkdir()
    from tau3_grpo.integrations.boundary_checkpoint import complete_boundary
    from tau3_grpo.training.rl.checkpoints import required_files
    for relative in required_files(4):
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    complete_boundary(tmp_path, 10, 4)
    (tmp_path / "swanlab-run.json").write_text("{}")
    command, _, _ = m.resolve(tmp_path, reward_version=version, reward_recipe=recipe,
                              resume_from=checkpoint)
    assert "trainer.resume_mode=resume_path" in command
    with pytest.raises(ValueError, match="dynamic filtering"):
        m.resolve(tmp_path, reward_version=version, reward_recipe=recipe,
                  resume_from=checkpoint, dynamic_filter=True)
    changed = deepcopy(report)
    changed["algorithm"] = {**ALGORITHM, "lambda_outcome": .5}
    new_recipe = tmp_path / "changed.json"
    new_recipe.write_text(json.dumps(freeze_recipe(changed)))
    with pytest.raises(ValueError, match="cannot change frozen"):
        m.resolve(tmp_path, reward_version=version, reward_recipe=new_recipe,
                  resume_from=checkpoint)
