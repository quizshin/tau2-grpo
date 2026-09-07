"""Dynamic Filtering: group classification, mask-only effect, d_bar telemetry."""

from __future__ import annotations

import numpy as np
import pytest

from tau3_grpo.algo.dynamic_filtering import (
    DEFAULT_GROUP_SIZE,
    FilterMode,
    GroupKind,
    apply_dynamic_filter,
    classify_group,
    evaluate_groups,
    group_indices,
    merge_stats,
    select_fixed_informative,
)

GROUP = DEFAULT_GROUP_SIZE


def _uids(*counts: tuple[str, int]) -> list[str]:
    out: list[str] = []
    for uid, count in counts:
        out.extend([uid] * count)
    return out


def test_all_zero_group_is_dropped():
    assert classify_group([0.0] * GROUP) is GroupKind.ALL_ZERO


def test_all_one_group_is_dropped():
    assert classify_group([1.0] * GROUP) is GroupKind.ALL_ONE


def test_mixed_group_is_informative():
    rewards = [1.0] + [0.0] * (GROUP - 1)
    assert classify_group(rewards) is GroupKind.INFORMATIVE


def test_uniform_nonzero_group_has_no_contrast():
    assert classify_group([0.5] * GROUP) is GroupKind.ALL_ONE


def test_wrong_size_group_is_incomplete():
    assert classify_group([1.0, 0.0]) is GroupKind.INCOMPLETE


def test_group_indices_preserves_first_seen_order():
    groups = group_indices(["b", "a", "b", "a"])
    assert list(groups) == ["b", "a"]
    assert groups["b"] == [0, 2]
    assert groups["a"] == [1, 3]


def test_evaluate_groups_counts_each_kind():
    rewards = [0.0] * GROUP + [1.0] * GROUP + ([1.0] + [0.0] * (GROUP - 1))
    uids = _uids(("zero", GROUP), ("one", GROUP), ("mixed", GROUP))
    stats = evaluate_groups(rewards, uids, group_size=GROUP)
    assert stats.candidate_groups == 3
    assert stats.effective_groups == 1
    assert stats.all_zero_groups == 1
    assert stats.all_one_groups == 1


def test_d_bar_and_inverse():
    rewards = [0.0] * GROUP + ([1.0] + [0.0] * (GROUP - 1))
    uids = _uids(("zero", GROUP), ("mixed", GROUP))
    stats = evaluate_groups(rewards, uids, group_size=GROUP)
    assert stats.d_bar == pytest.approx(0.5)
    assert stats.inverse_one_minus_d_bar == pytest.approx(2.0)


def test_d_bar_is_zero_when_all_informative():
    rewards = ([1.0] + [0.0] * (GROUP - 1)) * 2
    uids = _uids(("a", GROUP), ("b", GROUP))
    stats = evaluate_groups(rewards, uids, group_size=GROUP)
    assert stats.d_bar == 0.0
    assert stats.inverse_one_minus_d_bar == pytest.approx(1.0)


def test_inverse_is_inf_when_everything_dropped():
    stats = evaluate_groups([0.0] * GROUP, ["u"] * GROUP, group_size=GROUP)
    assert stats.d_bar == 1.0
    assert stats.inverse_one_minus_d_bar == float("inf")


def test_mask_is_zeroed_only_for_degenerate_groups():
    rewards = [0.0] * GROUP + ([1.0] + [0.0] * (GROUP - 1))
    uids = _uids(("zero", GROUP), ("mixed", GROUP))
    mask = np.ones((2 * GROUP, 5), dtype=np.int64)
    filtered, stats = apply_dynamic_filter(mask, rewards, uids, group_size=GROUP)

    assert filtered[:GROUP].sum() == 0
    assert filtered[GROUP:].sum() == GROUP * 5
    assert stats.effective_groups == 1


def test_filter_does_not_mutate_input_mask():
    rewards = [0.0] * GROUP
    uids = ["u"] * GROUP
    mask = np.ones((GROUP, 3), dtype=np.int64)
    filtered, _ = apply_dynamic_filter(mask, rewards, uids, group_size=GROUP)
    assert mask.sum() == GROUP * 3
    assert filtered.sum() == 0


def test_filter_preserves_batch_shape():
    rewards = [0.0] * GROUP
    uids = ["u"] * GROUP
    mask = np.ones((GROUP, 7), dtype=np.int64)
    filtered, _ = apply_dynamic_filter(mask, rewards, uids, group_size=GROUP)
    assert filtered.shape == mask.shape


def test_filter_rejects_length_mismatch():
    with pytest.raises(ValueError, match="mask rows"):
        apply_dynamic_filter(np.ones((3, 2)), [0.0, 1.0], ["a", "b"], group_size=2)


def test_evaluate_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        evaluate_groups([0.0, 1.0], ["a"], group_size=2)


def test_no_state_carries_across_updates():
    rewards_a = [0.0] * GROUP
    rewards_b = ([1.0] + [0.0] * (GROUP - 1))
    uids = ["u"] * GROUP
    first = evaluate_groups(rewards_a, uids, group_size=GROUP)
    second = evaluate_groups(rewards_b, uids, group_size=GROUP)
    assert first.candidate_groups == 1
    assert second.candidate_groups == 1
    assert first.effective_groups == 0
    assert second.effective_groups == 1


def test_fixed_informative_cap_is_disabled_by_default():
    rewards = ([1.0] + [0.0] * (GROUP - 1)) * 3
    uids = _uids(("a", GROUP), ("b", GROUP), ("c", GROUP))
    kept, stats = select_fixed_informative(
        rewards, uids, group_size=GROUP, target_informative_groups=3
    )
    assert kept == ["a", "b", "c"]
    assert stats.mode == FilterMode.FIXED_INFORMATIVE.value


def test_fixed_informative_respects_explicit_cap():
    rewards = ([1.0] + [0.0] * (GROUP - 1)) * 3
    uids = _uids(("a", GROUP), ("b", GROUP), ("c", GROUP))
    kept, _ = select_fixed_informative(
        rewards,
        uids,
        group_size=GROUP,
        target_informative_groups=3,
        max_candidate_groups=2,
    )
    assert kept == ["a", "b"]


def test_merge_stats_is_reporting_only():
    rewards = [0.0] * GROUP
    uids = ["u"] * GROUP
    stats = [evaluate_groups(rewards, uids, group_size=GROUP) for _ in range(3)]
    merged = merge_stats(stats)
    assert merged["updates"] == 3
    assert merged["candidate_groups"] == 3
    assert merged["effective_groups"] == 0
    assert merged["d_bar_mean"] == pytest.approx(1.0)
