import math

import pytest

from tau3_grpo.algorithms.arpo import (
    ARPOConfig,
    branch_probability,
    group_rows,
    partial_entropy,
    settings,
    stable_seed,
    worker_rows,
)


def test_entropy_is_partial_mass_not_surprisal_or_renormalized():
    got = partial_entropy([[math.log(.5), math.log(.25)]], vocab_size=4, window=20)
    assert got['value'] == pytest.approx(.5)
    assert got['tokens'] == 1
    assert sum(got['contributions']) / math.log(4) == got['value']
    with pytest.raises(ValueError):
        partial_entropy([], vocab_size=4, window=20)
    with pytest.raises(ValueError):
        partial_entropy([[0., 0.]], vocab_size=4, window=20)


def test_budget_partition_never_splits_groups_and_preserves_all_rows():
    uids = ['a', 'b', 'a', 'b', 'c', 'c']
    bins = worker_rows(uids, 2, 8)
    assert bins == [[0, 2], [1, 3], [4, 5]]
    assert sorted(sum(bins, [])) == list(range(6))
    with pytest.raises(ValueError, match='complete'):
        group_rows(['a', 'a', 'b'], 2)


def test_probability_and_seed():
    c = ARPOConfig()
    assert branch_probability(1, 0, c) == 1
    assert branch_probability(0, 10, c) == 0
    assert branch_probability(1, 1, c) == .5
    assert stable_seed(42, 'a', 1) == stable_seed(42, 'a', 1)
    assert stable_seed(42, 'a', 1) != stable_seed(42, 'a', 2)


@pytest.mark.parametrize('kwargs', [dict(initial_rollouts=0), dict(initial_rollouts=1.5),
    dict(entropy_mode='surprisal'), dict(advantage_mode='hard'), dict(entropy_weight=float('nan'))])
def test_bad_settings(kwargs):
    with pytest.raises(ValueError):
        ARPOConfig(**kwargs)


def test_no_silent_filtering_or_unknown_config():
    for c in ({'dynamic_filter': {'enable': True}}, {'arpo': {'typo': 1}},
              {'use_kl_in_reward': True}, {'norm_adv_by_std_in_grpo': False}):
        with pytest.raises(ValueError):
            settings(dict(adv_estimator='arpo', **c))
