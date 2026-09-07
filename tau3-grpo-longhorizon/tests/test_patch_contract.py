"""veRL local patch contract.

These assertions run on the vendored veRL tree without importing veRL, so they
work on CPU. A veRL upgrade that dropped a patch fails here instead of silently
producing wrong advantages.
"""

from __future__ import annotations

import pytest

from tau3_grpo.integration.patch_contract import (
    ALGORITHM_CONFIG,
    PATCH_MARKER,
    RAY_TRAINER,
    REQUIREMENTS,
    TOOL_AGENT_LOOP,
    assert_patched,
    check_patch,
    vanilla_grpo_untouched,
    verify_all,
)
from tau3_grpo.paths import VERL_ROOT


def test_verl_tree_present():
    assert VERL_ROOT.is_dir(), f"vendored veRL not found at {VERL_ROOT}"


def test_all_patches_applied():
    problems = verify_all()
    assert problems == [], "unsatisfied patch contracts:\n" + "\n".join(problems)


def test_assert_patched_passes():
    assert_patched()


@pytest.mark.parametrize("requirement", REQUIREMENTS, ids=lambda r: str(r.relative_path))
def test_each_requirement(requirement):
    assert check_patch(requirement) == []


def test_marker_present_in_every_patched_file():
    for requirement in REQUIREMENTS:
        text = (VERL_ROOT / requirement.relative_path).read_text(encoding="utf-8")
        assert PATCH_MARKER in text, f"{requirement.relative_path} lacks the patch marker"


def test_vanilla_grpo_path_untouched():
    assert vanilla_grpo_untouched(), (
        "the GRPO branch of compute_advantage was modified; the patch must leave "
        "the default path alone"
    )


def test_tool_agent_loop_aligns_none_at_observations():
    text = (VERL_ROOT / TOOL_AGENT_LOOP).read_text(encoding="utf-8")
    # Both observation handlers must record a placeholder so spans stay aligned.
    assert text.count("tau3_anchor_hook") >= 3, (
        "expected the anchor hook in the generating, tool and interacting handlers"
    )


def test_tool_agent_loop_publishes_official_terminal_reward():
    text = (VERL_ROOT / TOOL_AGENT_LOOP).read_text(encoding="utf-8")
    assert "finalize_rollout" in text
    assert "reward_score=terminal_reward_score" in text
    assert "reward_extra_info" in text


def test_tool_agent_loop_lazily_loads_worker_anchor_hook():
    text = (VERL_ROOT / TOOL_AGENT_LOOP).read_text(encoding="utf-8")
    assert "TAU3_GRPO_ANCHOR_HOOK" in text
    assert "_load_tau3_anchor_hook_from_env" in text


def test_tool_agent_loop_enforces_per_turn_generation_cap():
    text = (VERL_ROOT / TOOL_AGENT_LOOP).read_text(encoding="utf-8")
    assert "TAU3_GRPO_MAX_TOKENS_PER_TURN" in text
    assert 'turn_sampling_params["max_tokens"]' in text


def test_tool_agent_loop_records_parser_and_dispatch_errors_for_replay():
    text = (VERL_ROOT / TOOL_AGENT_LOOP).read_text(encoding="utf-8")
    assert "record_tool_failure" in text
    assert "tool arguments must decode to an object" in text
    assert '"dispatch_error": True' in text


def test_ray_trainer_gates_df_on_config():
    text = (VERL_ROOT / RAY_TRAINER).read_text(encoding="utf-8")
    assert "tau3_dynamic_filter" in text
    # DF must be opt-in so vanilla GRPO is unaffected.
    assert "dynamic_filter" in text


def test_ray_trainer_masks_and_removes_policy_dp_padding():
    text = (VERL_ROOT / RAY_TRAINER).read_text(encoding="utf-8")
    assert "tau3_pad_policy_batch" in text
    assert "tau3_unpad_policy_batch" in text
    assert '"response_mask", "rm_scores", "reward_scores"' in text
    assert "tau3_is_padding" in text
    assert "TAU3_GRPO_POLICY_BATCH_DIVISOR" in text


def test_ray_trainer_skips_all_filtered_actor_update_without_nan():
    text = (VERL_ROOT / RAY_TRAINER).read_text(encoding="utf-8")
    assert "tau3_has_effective_policy_tokens" in text
    assert "tau3_restore_candidate_mask_for_metrics" in text
    assert "dynamic_filter/skipped_actor_update" in text
    assert "actor/skipped_no_effective_tokens" in text


def test_ray_trainer_registers_estimator_inside_task_runner_process():
    text = (VERL_ROOT / RAY_TRAINER).read_text(encoding="utf-8")
    assert "register_tau3_gigpo" in text


def test_algorithm_config_defaults_to_none():
    text = (VERL_ROOT / ALGORITHM_CONFIG).read_text(encoding="utf-8")
    assert "dynamic_filter" in text
    assert "gigpo" in text
    # Optional means default None, so existing configs keep working.
    assert "Optional[dict[str, Any]] = None" in text or "= None" in text


def test_missing_file_is_reported(tmp_path):
    problems = verify_all(tmp_path)
    assert problems
    assert any("missing patched file" in problem for problem in problems)


def test_assert_patched_raises_on_empty_tree(tmp_path):
    with pytest.raises(RuntimeError, match="local patches are missing"):
        assert_patched(tmp_path)


@pytest.mark.verl
def test_tau_gigpo_registers():
    pytest.importorskip("verl", reason="veRL not installed")
    pytest.importorskip("torch", reason="torch not installed")
    from tau3_grpo.algo.verl_estimator import ESTIMATOR_NAME, is_registered, register

    register()
    assert is_registered()

    from verl.trainer.ppo.core_algos import get_adv_estimator_fn

    assert get_adv_estimator_fn(ESTIMATOR_NAME) is not None


@pytest.mark.verl
def test_register_is_idempotent():
    pytest.importorskip("verl", reason="veRL not installed")
    from tau3_grpo.algo.verl_estimator import register

    register()
    assert register() is False


@pytest.mark.verl
def test_anchor_hook_installs_and_uninstalls():
    pytest.importorskip("verl", reason="veRL not installed")
    from tau3_grpo.integration import anchor_hook

    anchor_hook.install(force=True)
    assert anchor_hook.is_installed()
    anchor_hook.uninstall()
    assert not anchor_hook.is_installed()


@pytest.mark.verl
def test_anchor_hook_is_inert_without_a_session():
    pytest.importorskip("verl", reason="veRL not installed")
    from tau3_grpo.integration.anchor_hook import current_anchor

    class _AgentData:
        request_id = "not-registered"

    assert current_anchor(_AgentData(), "assistant") is None


@pytest.mark.verl
def test_hook_records_none_for_observation_segments():
    pytest.importorskip("verl", reason="veRL not installed")
    from verl.experimental.agent_loop.tool_agent_loop import (
        set_tau3_anchor_hook,
        tau3_anchor_hook,
    )

    class _AgentData:
        def __init__(self):
            self.request_id = "r"
            self.anchor_ids = []
            self.anchor_spans = []

    set_tau3_anchor_hook(lambda agent_data, kind: "anchor-1")
    try:
        data = _AgentData()
        tau3_anchor_hook(data, "assistant", 0, 5)
        tau3_anchor_hook(data, "tool", 5, 9)
        tau3_anchor_hook(data, "user", 9, 12)
        assert data.anchor_ids == ["anchor-1", None, None]
        assert data.anchor_spans == [(0, 5), None, None]
    finally:
        set_tau3_anchor_hook(None)


@pytest.mark.verl
def test_hook_failure_does_not_break_rollout():
    pytest.importorskip("verl", reason="veRL not installed")
    from verl.experimental.agent_loop.tool_agent_loop import (
        set_tau3_anchor_hook,
        tau3_anchor_hook,
    )

    def _boom(agent_data, kind):
        raise RuntimeError("anchor resolution failed")

    class _AgentData:
        def __init__(self):
            self.request_id = "r"
            self.anchor_ids = []
            self.anchor_spans = []

    set_tau3_anchor_hook(_boom)
    try:
        data = _AgentData()
        tau3_anchor_hook(data, "assistant", 0, 5)
        assert data.anchor_ids == [None]
        assert data.anchor_spans == [None]
    finally:
        set_tau3_anchor_hook(None)


@pytest.mark.verl
def test_estimator_matches_pure_function():
    pytest.importorskip("verl", reason="veRL not installed")
    torch = pytest.importorskip("torch", reason="torch not installed")

    from tau3_grpo.algo.tau_gigpo import StepRecord, compute_tau_gigpo_advantage
    from tau3_grpo.algo.verl_estimator import compute_tau_gigpo_verl

    rewards = torch.zeros(2, 4)
    rewards[0, 3] = 1.0
    mask = torch.ones(2, 4)
    non_tensor = {
        "anchor_ids": [["a"], ["a"]],
        "anchor_spans": [[[0, 2]], [[0, 2]]],
    }
    advantages, _ = compute_tau_gigpo_verl(
        rewards, mask, index=["u", "u"], config=None, non_tensor_batch=non_tensor
    )
    expected, _ = compute_tau_gigpo_advantage(
        [1.0, 0.0],
        ["u", "u"],
        [StepRecord(0, 0, "a", (0, 2)), StepRecord(1, 0, "a", (0, 2))],
        response_length=4,
        response_mask=mask.numpy(),
    )
    assert advantages.numpy() == pytest.approx(expected)
