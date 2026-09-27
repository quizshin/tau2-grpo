import pytest

from tau3_grpo.data.sampling import sampling_identities


def test_group_crossing_worker_boundary_keeps_trial_numbers():
    ids = sampling_identities(["a"] * 8 + ["b"] * 8, data_seed=42, step=3)
    chunks = [ids[:5], ids[5:10], ids[10:]]
    assert [row["trial"] for chunk in chunks for row in chunk if row["sample_group_uid"] == "a"] == list(range(8))
    assert ids[8]["trial"] == 0
    assert all(row["generation_request_seed"] is None for row in ids)
    assert ids[0]["seed_semantics"] == "configured_data_seed"


def test_interleaved_groups_use_uid_and_not_adjacent_row_position():
    ids = sampling_identities(["a", "b", "a", "b"], data_seed=42, step=1, validation=True)
    assert [row["trial"] for row in ids] == [0, 0, 1, 1]
    assert all(row["validation"] for row in ids)
    assert ids == sampling_identities(["a", "b", "a", "b"], data_seed=42, step=1, validation=True)


@pytest.mark.parametrize("uid", [None, ""])
def test_missing_identity_is_not_invented(uid):
    with pytest.raises(ValueError):
        sampling_identities([uid], data_seed=None, step=1)
