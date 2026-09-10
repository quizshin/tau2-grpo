"""Tau-GiGPO: A = A_episode + omega*A_step, Fnorm=1, gamma=0.95, A_step=0 rule."""

from __future__ import annotations

import numpy as np
import pytest

from tau3_grpo.algorithms.tau_gigpo import (
    FNORM,
    GAMMA,
    MIN_ANCHOR_GROUP_SIZE,
    StepRecord,
    combine_advantages,
    compute_tau_gigpo_advantage,
    episode_advantages,
    step_advantages,
    step_returns,
    steps_from_anchor_payload,
)


def test_frozen_constants():
    assert GAMMA == 0.95
    assert FNORM == 1.0
    assert MIN_ANCHOR_GROUP_SIZE == 2


def test_episode_advantage_is_mean_centred_not_std_normalised():
    # Fnorm=1 means no division by group std.
    adv = episode_advantages([1.0, 0.0], ["u", "u"])
    assert adv[0] == pytest.approx(0.5)
    assert adv[1] == pytest.approx(-0.5)


def test_episode_advantage_sums_to_zero_per_group():
    adv = episode_advantages([1.0, 0.0, 0.0, 1.0], ["a", "a", "b", "b"])
    assert adv[:2].sum() == pytest.approx(0.0)
    assert adv[2:].sum() == pytest.approx(0.0)


def test_episode_advantage_groups_independently():
    adv = episode_advantages([1.0, 1.0, 0.0, 1.0], ["a", "a", "b", "b"])
    assert adv[0] == pytest.approx(0.0)
    assert adv[1] == pytest.approx(0.0)
    assert adv[2] == pytest.approx(-0.5)
    assert adv[3] == pytest.approx(0.5)


def test_step_return_discounts_by_remaining_steps():
    steps = [
        StepRecord(0, 0, "a", (0, 2)),
        StepRecord(0, 1, "b", (2, 4)),
        StepRecord(0, 2, "c", (4, 6)),
    ]
    returns = step_returns(steps, [1.0], gamma=GAMMA)
    assert returns[2] == pytest.approx(1.0)
    assert returns[1] == pytest.approx(GAMMA)
    assert returns[0] == pytest.approx(GAMMA**2)


def test_step_advantage_zero_when_no_group_reaches_two():
    steps = [StepRecord(0, 0, "solo_a", (0, 2)), StepRecord(1, 0, "solo_b", (0, 2))]
    adv, stats = step_advantages(steps, [1.0, 0.0])
    assert np.all(adv == 0.0)
    assert stats.usable_anchor_groups == 0
    assert stats.step_advantage_active is False


def test_step_advantage_active_when_group_has_two():
    steps = [StepRecord(0, 0, "shared", (0, 2)), StepRecord(1, 0, "shared", (0, 2))]
    adv, stats = step_advantages(steps, [1.0, 0.0])
    assert stats.usable_anchor_groups == 1
    assert stats.step_advantage_active is True
    assert adv[0] == pytest.approx(0.5)
    assert adv[1] == pytest.approx(-0.5)


def test_singleton_groups_stay_zero_alongside_a_usable_group():
    steps = [
        StepRecord(0, 0, "shared", (0, 2)),
        StepRecord(1, 0, "shared", (0, 2)),
        StepRecord(2, 0, "solo", (0, 2)),
    ]
    adv, stats = step_advantages(steps, [1.0, 0.0, 1.0])
    assert stats.usable_anchor_groups == 1
    assert adv[2] == 0.0


def test_unanchored_steps_are_ignored():
    steps = [StepRecord(0, 0, None, (0, 2)), StepRecord(1, 0, None, (0, 2))]
    adv, stats = step_advantages(steps, [1.0, 0.0])
    assert np.all(adv == 0.0)
    assert stats.anchored_steps == 0
    assert stats.anchor_coverage == 0.0


def test_combine_adds_omega_weighted_step_only_on_its_span():
    episode = np.array([0.5])
    step = np.array([1.0])
    steps = [StepRecord(0, 0, "a", (1, 3))]
    out = combine_advantages(
        episode, step, steps, num_trajectories=1, response_length=5, omega=0.5
    )
    assert out[0, 0] == pytest.approx(0.5)
    assert out[0, 1] == pytest.approx(1.0)
    assert out[0, 2] == pytest.approx(1.0)
    assert out[0, 3] == pytest.approx(0.5)


def test_combine_respects_response_mask():
    episode = np.array([1.0])
    steps = [StepRecord(0, 0, "a", (0, 2))]
    mask = np.array([[1, 0, 1, 0]])
    out = combine_advantages(
        episode, np.array([0.0]), steps, num_trajectories=1, response_length=4,
        response_mask=mask, omega=0.5,
    )
    assert out[0, 1] == 0.0
    assert out[0, 3] == 0.0
    assert out[0, 0] == pytest.approx(1.0)


def test_combine_clips_span_to_response_length():
    steps = [StepRecord(0, 0, "a", (2, 99))]
    out = combine_advantages(
        np.array([0.0]), np.array([1.0]), steps,
        num_trajectories=1, response_length=4, omega=1.0,
    )
    assert out[0, 2] == pytest.approx(1.0)
    assert out.shape == (1, 4)


def test_full_formula_equals_episode_plus_omega_step():
    steps = [StepRecord(0, 0, "shared", (0, 2)), StepRecord(1, 0, "shared", (0, 2))]
    adv, stats = compute_tau_gigpo_advantage(
        [1.0, 0.0], ["u", "u"], steps, response_length=2, omega=0.5
    )
    # A_episode = +/-0.5, A_step = +/-0.5, omega=0.5 -> 0.5 + 0.25
    assert adv[0, 0] == pytest.approx(0.75)
    assert adv[1, 0] == pytest.approx(-0.75)
    assert stats.step_advantage_active is True


def test_degrades_to_episode_grpo_without_anchor_groups():
    steps = [StepRecord(0, 0, "solo_a", (0, 2)), StepRecord(1, 0, "solo_b", (0, 2))]
    adv, stats = compute_tau_gigpo_advantage(
        [1.0, 0.0], ["u", "u"], steps, response_length=2, omega=0.5
    )
    assert adv[0, 0] == pytest.approx(0.5)
    assert adv[1, 0] == pytest.approx(-0.5)
    assert stats.step_advantage_active is False


def test_omega_zero_removes_step_term():
    steps = [StepRecord(0, 0, "shared", (0, 2)), StepRecord(1, 0, "shared", (0, 2))]
    adv, _ = compute_tau_gigpo_advantage(
        [1.0, 0.0], ["u", "u"], steps, response_length=2, omega=0.0
    )
    assert adv[0, 0] == pytest.approx(0.5)


def test_returns_uids_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        episode_advantages([1.0, 0.0], ["u"])


def test_nonpositive_fnorm_raises():
    with pytest.raises(ValueError, match="fnorm must be positive"):
        episode_advantages([1.0, 0.0], ["u", "u"], fnorm=0.0)


def test_steps_from_anchor_payload_skips_none_segments():
    ids = [["a", None, "b", None]]
    spans = [[[0, 2], None, [4, 6], None]]
    steps = steps_from_anchor_payload(ids, spans)
    assert len(steps) == 2
    assert [step.step_index for step in steps] == [0, 1]
    assert steps[0].span == (0, 2)
    assert steps[1].anchor_id == "b"


def test_steps_from_anchor_payload_handles_empty_rows():
    steps = steps_from_anchor_payload([[], []], [[], []])
    assert steps == []


def test_step_index_is_per_trajectory():
    ids = [["a", "b"], ["a"]]
    spans = [[[0, 1], [1, 2]], [[0, 1]]]
    steps = steps_from_anchor_payload(ids, spans)
    assert [(s.trajectory_index, s.step_index) for s in steps] == [(0, 0), (0, 1), (1, 0)]
