import json

import pytest

from tau3_grpo.data.trajectory import trajectory_facts
from tau3_grpo.tracking.trainer_telemetry import collect_rollout_metrics


def facts(**changes):
    values = dict(request_id="r", task_id="t", turns=[{"token_span": [0, 2]}, {"token_span": [3, 6]}],
                  response_ids=[10, 11, 12, 13], response_mask=[1, 1, 0, 1])
    return trajectory_facts(**(values | changes))


def test_span_clipping_keeps_emitted_and_retained_evidence():
    result = facts(response_logprobs=[-.2])
    assert result["turns"][1]["token_span"] == [3, 6]
    assert result["turns"][1]["retained_token_span"] == [3, 4]
    assert result["turns"][1]["tokens_discarded"] == 2
    assert result["tokens"]["response_logprobs"] is None
    assert not result["capabilities"]["complete_response_logprobs"]


@pytest.mark.parametrize("spans", [[[0, 4]], [[0, 1], [3, 4]], [[0, 2], [1, 2], [3, 4]]])
def test_invalid_token_attribution_fails_closed(spans):
    with pytest.raises(ValueError):
        facts(turns=[{"token_span": s} for s in spans])


def test_telemetry_does_not_count_padding_or_stale_global_stats(monkeypatch):
    from tau3_grpo.integrations.verl import gigpo

    monkeypatch.setattr(gigpo, "_LAST_STATS", {"wrong_batch": 1000})
    result = collect_rollout_metrics(
        {"trajectory_facts_json": [json.dumps(facts()), json.dumps(facts())],
         "tau3_is_padding": [False, True]}, adv_estimator="tau_gigpo",
        estimator_diagnostics={"estimator": "tau_gigpo", "stats": {"current": 2}},
    )
    assert result["gigpo/current"] == 2
    assert "gigpo/wrong_batch" not in result
    assert result["rollout/facts_trajectories"] == 1
    assert result["rollout/exact_generated_tokens"] == 3
    assert result["rollout/raw_finish_reason/unavailable"] == 2
    with pytest.raises(ValueError, match="different algorithm"):
        collect_rollout_metrics({}, adv_estimator="grpo",
                                estimator_diagnostics={"estimator": "tau_gigpo"})


def test_logprob_length_alone_does_not_prove_alignment_after_truncation():
    turns = [{"token_span": [0, 2], "generated_logprobs_available": False},
             {"token_span": [3, 6], "generated_logprobs_available": True}]
    result = facts(turns=turns, response_logprobs=[-.2, -.3, 0., -.4])
    assert result["tokens"]["response_logprobs"] is None
    turns[0]["generated_logprobs_available"] = True
    result = facts(turns=turns, response_logprobs=[-.2, -.3, 0., -.4])
    assert result["tokens"]["response_logprobs"] == [-.2, -.3, 0., -.4]
    assert result["capabilities"]["complete_response_logprobs"]
