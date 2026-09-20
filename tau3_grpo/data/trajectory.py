"""Algorithm-independent rollout facts; no reward shaping or token reconstruction."""
from __future__ import annotations

from copy import deepcopy


def trajectory_facts(*, request_id, task_id, turns, response_ids, response_mask,
                     response_logprobs=None, terminal=None, sample_group_uid=None,
                     trial=None, seed=None):
    ids, mask = list(response_ids), list(response_mask)
    if len(ids) != len(mask) or any(v not in (0, 1) for v in mask):
        raise ValueError("Trajectory token IDs and binary response mask must align")
    retained_turns = deepcopy(turns)
    coverage = [0] * len(ids)
    for turn in retained_turns:
        start, end = turn["token_span"]
        if (type(start) is not int or type(end) is not int or not 0 <= start <= end):
            raise ValueError("Invalid emitted assistant token span")
        retained = [min(start, len(ids)), min(end, len(ids))]
        turn["retained_token_span"] = retained
        turn["tokens_discarded"] = (end - start) - (retained[1] - retained[0])
        for index in range(*retained):
            if coverage[index] or not mask[index]:
                raise ValueError("Assistant spans overlap or include observation tokens")
            coverage[index] = 1
    if coverage != mask:
        raise ValueError("Assistant spans do not cover retained generated tokens")
    probabilities = None if response_logprobs is None else list(response_logprobs)
    complete_logprobs = (probabilities is not None and len(probabilities) == len(ids)
                         and all(t.get("generated_logprobs_available", False) for t in turns))
    return {
        "schema": "tau3_trajectory_facts_v1",
        "identity": {"trajectory_id": request_id, "task_id": task_id,
                     "sample_group_uid": sample_group_uid, "trial": trial, "seed": seed},
        "turns": retained_turns,
        "tokens": {"response_ids": ids, "response_mask": mask,
                   "response_logprobs": probabilities if complete_logprobs else None,
                   "logprob_semantics": "generated_tokens_only_observation_positions_are_placeholders"},
        "capabilities": {"exact_response_ids": True, "exact_response_mask": True,
                         "complete_response_logprobs": complete_logprobs,
                         "raw_finish_reason_turns": sum(t.get("finish_reason") is not None for t in turns)},
        "terminal": deepcopy(terminal or {}),
    }
