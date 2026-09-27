import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from tau3_grpo.algorithms.mt_gtpo_verl import compute_mt_gtpo_verl, last_filter_stats
from tau3_grpo.analysis.calibrate_process_rewards import analyze_update, summarize
from tau3_grpo.evaluation.process_reward import payload_json, score_turns
from tau3_grpo.launch import prepare
from tau3_grpo.paths import CODE_ROOT


def fixture_batch(
    outcomes=(0.0, 0.0, 0.0, 0.0), *, mode="conservative", errors=(False, True, False, False)
):
    payloads = []
    for error in errors:
        records = [
            {
                "schema": "tau3_turn_v1",
                "turn_index": 0,
                "token_span": [0, 2],
                "tool_calls": [
                    {"name": "calculate", "arguments": {"expression": "1+1"}, "error": error}
                ],
            }
        ]
        payloads.append(payload_json(score_turns(records, [], [], {"mode": mode})))
    return (
        torch.tensor(outcomes)[:, None],
        torch.ones((4, 2)),
        {
            "process_reward_json": np.array(payloads, dtype=object),
            "uid": np.array(["u", "u", "v", "v"]),
        },
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_verl_adapter_filter_toggle_and_roundtrip(enabled):
    rewards, mask, metadata = fixture_batch()
    config = {
        "process_reward": {"mode": "conservative"},
        "dynamic_filter": {"enable": enabled, "group_size": 2, "metric": "advantage"},
    }
    adv, returns = compute_mt_gtpo_verl(rewards, mask, metadata["uid"], config, metadata)
    assert adv[0, 0] > 0 and adv[1, 0] < 0
    assert mask.sum() == (4 if enabled else 8)
    assert returns[1, 0] == pytest.approx(-0.1)
    rows = [{"mt_gtpo_replay_json": raw} for raw in metadata["mt_gtpo_replay_json"]]
    observations = analyze_update(rows)
    report = summarize(observations)
    assert report["tiers"]["error"]["correlation"] is None  # all outcomes 0
    assert bool(last_filter_stats()) is enabled


def test_padding_is_not_a_group_member_or_advantage_peer():
    rewards, mask, metadata = fixture_batch()
    mask[2:] = 0
    metadata["tau3_is_padding"] = np.array([False, False, True, True])
    config = {
        "process_reward": {"mode": "conservative"},
        "dynamic_filter": {"enable": True, "group_size": 2},
    }
    adv, _ = compute_mt_gtpo_verl(rewards, mask, metadata["uid"], config, metadata)
    assert not adv[2:].any()
    assert last_filter_stats()["dynamic_filter/candidate_groups"] == 1


def test_ray_trainer_post_filter_all_drop_and_restore():
    from tensordict import TensorDict
    from verl.trainer.ppo.ray_trainer import (
        compute_advantage,
        tau3_dynamic_filter,
        tau3_has_effective_policy_tokens,
        tau3_restore_candidate_mask_for_metrics,
    )

    from verl import DataProto

    rewards, mask, metadata = fixture_batch(mode="audit")
    data = DataProto(
        batch=TensorDict(
            {"token_level_rewards": rewards.expand(-1, 2).clone(), "response_mask": mask},
            batch_size=[4],
        ),
        non_tensor_batch=metadata,
    )
    config = OmegaConf.create(
        {
            "adv_estimator": "mt_gtpo",
            "process_reward": {"mode": "audit"},
            "dynamic_filter": {"enable": True, "group_size": 2},
        }
    )
    assert tau3_dynamic_filter(data, config) == {}  # not filtered before normalization
    compute_advantage(data, "mt_gtpo", config=config)
    assert not tau3_has_effective_policy_tokens(data)
    assert not data.batch["advantages"].any()
    assert tau3_restore_candidate_mask_for_metrics(data, skipped=True)
    assert data.batch["response_mask"].sum() == 8
    assert not data.batch["advantages"].any()


def test_worker_trainer_config_drift_is_error():
    rewards, mask, metadata = fixture_batch()
    with pytest.raises(ValueError, match="mismatch"):
        compute_mt_gtpo_verl(rewards, mask, metadata["uid"], {}, metadata)
    with pytest.raises(ValueError, match="KL"):
        compute_mt_gtpo_verl(rewards, mask, metadata["uid"], {"use_kl_in_reward": True}, metadata)


@pytest.mark.parametrize("enabled", [False, True])
def test_v2_official_success_gate_adapter_replay_and_tamper(enabled):
    gold = {"name": "cancel_reservation", "arguments": {"reservation_id": "A"}}
    v2 = {"mode": "reference_write", "version": "v2"}
    turns = [{"schema": "tau3_turn_v1", "turn_index": 0, "token_span": [0, 1],
              "tool_calls": [{**gold, "error": False}]}]
    outcomes = (1., 0., 0., 0.)
    metadata = {"uid": np.array(["u", "u", "v", "v"]), "process_reward_json": np.array([
        payload_json(score_turns(turns, [gold], ["DB"], v2, official_outcome=o))
        for o in outcomes
    ], dtype=object)}
    rewards, mask = torch.tensor(outcomes)[:, None], torch.ones((4, 1))
    config = {"process_reward": v2, "dynamic_filter": {"enable": enabled, "group_size": 2}}
    adv, _ = compute_mt_gtpo_verl(rewards, mask, metadata["uid"], config, metadata)
    assert adv[0, 0] > 0 and adv[1, 0] < 0
    rows = [{"mt_gtpo_replay_json": r} for r in metadata["mt_gtpo_replay_json"]]
    assert len(analyze_update(rows)) == 4
    bad = json.loads(metadata["process_reward_json"][0])
    bad["official_outcome"] = 0
    metadata["process_reward_json"][0] = json.dumps(bad)
    with pytest.raises(ValueError, match="outcome mismatch"):
        compute_mt_gtpo_verl(rewards, mask, metadata["uid"], config, metadata)


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('recipe', [
    {'mode': 'reference_write', 'version': 'v3'},
    {'mode': 'paper', 'version': 'paper_v1'},
    {'mode': 'paper', 'version': 'paper_env_v2'},
])
def test_v3_and_paper_all_failure_group_keeps_partial_progress_and_replays(enabled, recipe):
    gold = {'name': 'cancel_reservation', 'arguments': {'reservation_id': 'A'}}
    payloads = []
    for matched in (True, False, False, False):
        call = {**gold, 'arguments': {'reservation_id': 'A' if matched else 'B'}, 'error': False}
        turns = [{'schema': 'tau3_turn_v1', 'turn_index': 0, 'token_span': [0, 1], 'tool_calls': [call]}]
        payloads.append(payload_json(score_turns(turns, [gold], ['DB'], recipe, official_outcome=0)))
    metadata = {'uid': np.array(['u', 'u', 'v', 'v']), 'process_reward_json': np.array(payloads, dtype=object)}
    mask = torch.ones((4, 1))
    config = {'process_reward': recipe, 'dynamic_filter': {'enable': enabled, 'group_size': 2}}
    adv, _ = compute_mt_gtpo_verl(torch.zeros((4, 1)), mask, metadata['uid'], config, metadata)
    assert adv[0, 0] > 0 and adv[1, 0] < 0
    assert torch.equal(mask[:, 0], torch.tensor([1., 1., 0., 0.] if enabled else [1., 1., 1., 1.]))
    assert len(analyze_update([{'mt_gtpo_replay_json': r} for r in metadata['mt_gtpo_replay_json']])) == 4
    bad = json.loads(metadata['process_reward_json'][0])
    bad['official_outcome'] = 1
    metadata['process_reward_json'][0] = json.dumps(bad)
    with pytest.raises(ValueError, match='outcome mismatch'):
        compute_mt_gtpo_verl(torch.zeros((4, 1)), mask, metadata['uid'], config, metadata)


@pytest.mark.parametrize(
    "profile,enabled", [("smoke", False), ("conservative", False), ("conservative_df", True)]
)
def test_launch_to_shell_and_hydra(profile, enabled, tmp_path):
    from hydra import compose, initialize_config_dir

    config = CODE_ROOT / f"configs/train/rl/qwen35_4b_lora_mt_gtpo_{profile}.yaml"
    env = {
        **os.environ,
        "TAU3_DRY_RUN": "1",
        "TAU3_ENV_FILE": str(tmp_path / "absent"),
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
    }
    command, env, _ = prepare("rl", config, "mt_gtpo", 42, [], env)
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, check=True, timeout=20
    )
    import shlex

    rendered = shlex.split(result.stdout.splitlines()[-1])
    assert "algorithm.adv_estimator=mt_gtpo" in rendered
    overrides = [s for s in rendered if s.startswith(("algorithm.", "+algorithm.", "++algorithm."))]
    with initialize_config_dir(
        config_dir=str(CODE_ROOT / "verl/verl/trainer/config"), version_base=None
    ):
        resolved = compose(config_name="ppo_trainer", overrides=overrides)
    assert resolved.algorithm.dynamic_filter.enable is enabled
    assert resolved.algorithm.dynamic_filter.metric == "advantage"
    assert resolved.algorithm.mt_gtpo.gamma == 0.9
    assert resolved.algorithm.process_reward.mode == (
        "audit" if profile == "smoke" else "conservative"
    )


def test_wrong_experiment_is_rejected():
    with pytest.raises(ValueError, match="experiment mt_gtpo"):
        prepare(
            "rl", CODE_ROOT / "configs/train/rl/qwen35_4b_lora_mt_gtpo_smoke.yaml", "e0", 42, [], {}
        )


def test_irc_cli_replay_and_reject_partial_or_tampered_update(tmp_path):
    from tau3_grpo.analysis.calibrate_process_rewards import main

    rewards, mask, metadata = fixture_batch(outcomes=(1.0, 0.0, 1.0, 0.0))
    compute_mt_gtpo_verl(
        rewards, mask, metadata["uid"], {"process_reward": {"mode": "conservative"}}, metadata
    )
    rows = [{"mt_gtpo_replay_json": raw} for raw in metadata["mt_gtpo_replay_json"]]
    path = tmp_path / "update.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    output = tmp_path / "report.json"
    assert main(["--input", str(path), "--output", str(output), "--min-support", "1"]) == 0
    report = json.loads(output.read_text())
    assert report["trajectories"] == 4
    assert report["tiers"]["error"]["correlation"] < 0
    with pytest.raises(ValueError, match="incomplete"):
        analyze_update(rows[1:])
    bad = json.loads(rows[0]["mt_gtpo_replay_json"])
    bad["process"]["turn_rewards"][0] = 123
    with pytest.raises(ValueError, match="reproduced"):
        analyze_update([{"mt_gtpo_replay_json": json.dumps(bad)}, *rows[1:]])


def test_df_cli_override_and_full_profile(tmp_path):
    import shlex

    from hydra import compose, initialize_config_dir

    path = CODE_ROOT / "configs/train/rl/qwen35_4b_full_mt_gtpo_candidate.yaml"
    command, env, _ = prepare(
        "rl",
        path,
        "mt_gtpo",
        42,
        ["algorithm.dynamic_filter.enable=true"],
        {
            **os.environ,
            "TAU3_DRY_RUN": "1",
            "TAU3_ENV_FILE": str(tmp_path / "absent"),
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        },
    )
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, check=True, timeout=20
    )
    rendered = shlex.split(result.stdout.splitlines()[-1])
    overrides = [s for s in rendered if s.startswith(("algorithm.", "+algorithm.", "++algorithm."))]
    with initialize_config_dir(
        config_dir=str(CODE_ROOT / "verl/verl/trainer/config"), version_base=None
    ):
        resolved = compose(config_name="ppo_trainer", overrides=overrides)
    assert resolved.algorithm.dynamic_filter.enable is True
    assert "actor_rollout_ref.model.lora_rank=0" in rendered


def test_mt_gtpo_explicit_diagnostics_are_owned_by_each_batch():
    from tau3_grpo.integrations.verl.mt_gtpo import last_stats

    rewards, mask, metadata = fixture_batch()
    config = {'process_reward': {'mode': 'conservative'},
              'dynamic_filter': {'enable': True, 'group_size': 2}}
    diagnostics = {}
    compute_mt_gtpo_verl(rewards, mask, metadata['uid'], config, metadata,
                         diagnostics_out=diagnostics)
    assert diagnostics['estimator'] == 'mt_gtpo'
    assert diagnostics['stats'] == last_stats()
    assert diagnostics['filter_stats'] == last_filter_stats()
    saved = json.dumps(diagnostics, sort_keys=True)
    rewards, mask, metadata = fixture_batch()
    compute_mt_gtpo_verl(rewards, mask, metadata['uid'],
                         {'process_reward': {'mode': 'conservative'}}, metadata,
                         diagnostics_out={})
    assert json.dumps(diagnostics, sort_keys=True) == saved
    assert last_filter_stats() == {}
