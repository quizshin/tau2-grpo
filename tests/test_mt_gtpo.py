import numpy as np
import pytest

from tau3_grpo.algorithms.dynamic_filtering import apply_advantage_filter
from tau3_grpo.algorithms.mt_gtpo import compute_mt_gtpo


def test_discount_and_observation_mask():
    mask = np.array([[1, 1, 0, 1], [1, 1, 0, 1]])
    adv, ret, detail = compute_mt_gtpo(
        [1, 0], ["u", "u"], [[1, 2], [0, 0]], [[[0, 2], [3, 4]]] * 2, mask, gamma=0.5
    )
    # G1=2+.5*1; G0=1+.5*G1. Terminal counted once at virtual K.
    np.testing.assert_allclose(ret[0], [2.25, 2.25, 0, 2.5])
    assert (adv[:, 2] == 0).all()
    np.testing.assert_allclose(detail["episode_advantages"], [0.999998, -0.999998], atol=1e-6)
    assert adv[0, 0] > 1 and adv[1, 0] < -1


def test_missing_late_turn_is_not_a_zero_reward_peer():
    mask = [[1, 0], [1, 1]]
    _, _, detail = compute_mt_gtpo(
        [0, 0], ["u", "u"], [[0], [0, 1]], [[[0, 1]], [[0, 1], [1, 2]]], mask
    )
    assert detail["turn_advantages"][0][0] < 0
    assert detail["turn_advantages"][1][0] > 0
    assert detail["turn_advantages"][1][1] == 0  # singleton


def test_same_task_different_uid_not_merged():
    adv, _, _ = compute_mt_gtpo(
        [1, 0], ["sample1", "sample2"], [[1], [0]], [[[0, 1]]] * 2, [[1], [1]]
    )
    assert (adv == 0).all()


def test_padding_and_empty_batch():
    adv, _, d = compute_mt_gtpo(
        [1, 0, 999], ["u"] * 3, [[0], [0], []], [[[0, 1]], [[0, 1]], []], [[1], [1], [0]]
    )
    assert d["active_rows"] == 2 and adv[2, 0] == 0
    empty, _, d = compute_mt_gtpo([0], ["u"], [[]], [[]], [[0]])
    assert empty[0, 0] == 0 and d["turns"] == 0


@pytest.mark.parametrize(
    "spans,mask",
    [
        ([[[0, 1]]], [[1, 1]]),
        ([[[0, 2]]], [[1, 0]]),
        ([[[0, 3]]], [[1, 1]]),
        ([[[1, 0]]], [[1, 1]]),
    ],
)
def test_invalid_span_rejected(spans, mask):
    with pytest.raises(ValueError):
        compute_mt_gtpo([1], ["u"], [[0]], spans, mask)


@pytest.mark.parametrize("outcome", [0, 1])
def test_filter_keeps_uniform_outcomes_with_process_signal(outcome):
    mask = np.ones((4, 1))
    adv, _, _ = compute_mt_gtpo(
        [outcome] * 4, ["u", "u", "v", "v"], [[1], [0], [0], [0]], [[[0, 1]]] * 4, mask
    )
    filtered, decisions, stats = apply_advantage_filter(
        mask, adv, ["u", "u", "v", "v"], group_size=2
    )
    assert decisions == {"u": True, "v": False}
    assert filtered.sum() == 2 and mask.sum() == 4
    assert stats["effective_groups"] == 1


def test_all_filtered_and_padding():
    filtered, decisions, _ = apply_advantage_filter(
        np.ones((3, 1)),
        np.zeros((3, 1)),
        ["u", "u", "dummy"],
        group_size=2,
        padding=[False, False, True],
    )
    assert not filtered.any() and decisions == {"u": False}
    with pytest.raises(ValueError, match="incomplete"):
        apply_advantage_filter([[1]], [[1]], ["u"], group_size=2)


def test_all_success_different_lengths_can_still_have_signal():
    _, _, d = compute_mt_gtpo(
        [1, 1], ["u", "u"], [[0], [0, 0]], [[[0, 1]], [[0, 1], [1, 2]]], [[1, 0], [1, 1]]
    )
    assert d["episode_advantages"] == [0, 0]
    assert d["nonzero_turns"] == 2
