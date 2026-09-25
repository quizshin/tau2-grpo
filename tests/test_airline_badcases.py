"""Badcase observations, raw-format adapters and immutable report contracts."""
import json
from copy import deepcopy

import pytest

from tau3_grpo.analysis.airline_badcases import (
    build_bundle,
    main,
    normalize_row,
    summarize,
    task_groups,
    verify_bundle,
)
from tau3_grpo.evaluation.eligibility import FALLBACK_TERMINATIONS, execution_eligibility
from tau3_grpo.utils.hashing import sha256_file


def rollout(**kw):
    return {"task_id": "task", "score": 0.0, "termination_reason": "user_stop",
            "scored": True, "trajectory_facts_json": {"turns": [], "identity": {}}, **kw}


def test_nested_evaluation_reads_actual_messages_and_nested_function_arguments():
    row = {"task_id": "task", "reward": 1, "termination_reason": "user_stop", "simulation": {
        "messages": [{"role": "assistant", "tool_calls": [
            {"function": {"name": "get_user_details", "arguments": '{"user_id":"u"}'}},
            {"name": "update_reservation_baggages", "arguments": {"id": 1}}]}]}}
    result = normalize_row(row, "SFT", sft=True)
    assert result["tool_call_count"] == 2 and result["read_call_count"] == 1
    assert result["write_call_count"] == 1 and result["success"] is True
    assert result["tool_evidence_source"] == "simulation.messages"
    different = deepcopy(row)
    different["simulation"]["messages"][0]["tool_calls"] = [
        {"function": {"name": "get_user_details", "arguments": '{"user_id":"u"}'}},
        {"function": {"name": "get_user_details", "arguments": '{"user_id":"v"}'}}]
    assert "repeated_call_signature" not in normalize_row(different, "SFT")["badcase_labels"]


def test_nested_tool_observation_errors_are_not_simulator_failures():
    row = {"task_id": "task", "reward": 0, "termination_reason": "user_stop", "simulation": {
        "messages": [{"role": "assistant", "tool_calls": [
            {"name": "book_reservation", "arguments": {}, "id": "call1"}]},
            {"role": "tool", "id": "call1", "error": True, "content": "missing required amount"}]}}
    result = normalize_row(row, "SFT")
    assert "tool_execution_error" in result["badcase_labels"]
    assert "infrastructure_error" not in result["badcase_labels"]
    assert result["success"] is False


def test_facts_take_precedence_and_counts_are_not_structured_calls():
    row = rollout(trajectory_json={"tool_calls": 99})
    assert normalize_row(row, "GRPO")["tool_call_count"] == 0
    del row["trajectory_facts_json"]
    result = normalize_row(row, "GRPO")
    assert result["tool_call_count"] is None
    assert "no_tool_calls_observed_on_failure" not in result["badcase_labels"]
    assert summarize([result])["sources"]["GRPO"]["tool_telemetry_missing_records"] == 1


@pytest.mark.parametrize("failure,term,label", [
    ("tool_errors", "user_stop", "tool_execution_error"),
    ("agent_error", "context_window_exceeded", "agent_generation_failure"),
    ("agent_error", "agent_error", "agent_generation_failure"),
])
def test_model_errors_remain_scored_failures(failure, term, label):
    result = normalize_row(rollout(failure_category=failure, termination_reason=term), "GRPO")
    assert result["success"] is False and label in result["badcase_labels"]
    assert task_groups([result])["task"]["bucket"] == "stable_failure"


@pytest.mark.parametrize("change", [
    {"termination_reason": "infrastructure_error"}, {"failure_category": "infrastructure"},
    {"score": None}, {"scored": False}, {"score": 0, "error": "endpoint unavailable"},
    {"execution_eligibility": {"officially_scored": False}},
    {"termination_reason": "future_unknown_reason"},
])
def test_unscored_is_unknown_and_kept_in_denominator(change):
    result = normalize_row(rollout(**change), "GRPO")
    assert result["success"] is None
    assert task_groups([result])["task"]["bucket"] == "environment_or_unscorable"
    good = normalize_row(rollout(score=1), "GRPO")
    summary = summarize([result, good])["sources"]["GRPO"]
    assert summary["rollouts"] == 2 and summary["successes"] == 1 and summary["unresolved"] == 1
    assert summary["success_rate"] is None


@pytest.mark.parametrize("reason", sorted(FALLBACK_TERMINATIONS))
@pytest.mark.parametrize("nested", [False, True])
def test_real_verifier_unscored_fallbacks_remain_complete_zero_trials(reason, nested):
    eligibility = execution_eligibility(reason, reward=0)
    row = rollout(termination_reason=reason, scored=False)
    if nested:
        row["trajectory_facts_json"]["terminal"] = {"scored": False, "execution_eligibility": eligibility}
    else:
        row["execution_eligibility"] = eligibility
    result = normalize_row(row, "GRPO")
    assert not result["scored"] and result["evaluation_trial_complete"] and result["success"] is False
    assert task_groups([result])["task"]["bucket"] == "stable_failure"
    summary = summarize([result, normalize_row(rollout(score=1), "GRPO")])["sources"]["GRPO"]
    assert summary["success_rate"] == .5 and summary["unresolved"] == 0
    assert summary["officially_scored"] == 1 and summary["completed_fallback_zeros"] == 1


def test_explicit_unresolved_and_unknown_termination_cannot_be_promoted():
    for change in [{"termination_reason": "max_steps", "execution_eligibility": {"evaluation_trial_complete": False}},
                   {"termination_reason": "unknown", "execution_eligibility": {"evaluation_trial_complete": True}}]:
        assert normalize_row(rollout(**change), "raw")["success"] is None
    with pytest.raises(ValueError, match="positive official outcome"):
        normalize_row(rollout(termination_reason="max_steps", score=1), "raw")


def test_partial_reward_is_not_success_and_demo_not_scored():
    assert normalize_row(rollout(score=.5), "GRPO")["success"] is False
    assert normalize_row(rollout(score=1 - 1e-7), "GRPO")["success"] is True
    demo = normalize_row({"messages": [{"role": "assistant", "content": "done"}], "reward": 1}, "SFT", kind="demonstration")
    assert demo["success"] is None and not task_groups([demo])


def test_attempts_and_hash_changes_never_become_correctness_labels():
    row = rollout(initial_db_hash="before", db_hash="after", trajectory_facts_json={"turns": [{"tool_calls": [
        {"name": "cancel_reservation", "arguments": {"id": "wrong"}, "error": True}]}]})
    result = normalize_row(row, "GRPO")
    assert set(result["badcase_labels"]) == {"db_changed_on_failure", "write_attempt_on_failure",
                                            "write_attempt_without_later_read", "tool_execution_error"}


def test_malformed_and_lossy_inputs_fail_closed():
    with pytest.raises(ValueError):
        normalize_row(rollout(trajectory_facts_json="{broken"), "GRPO")
    with pytest.raises(ValueError, match="normalized"):
        normalize_row({"record_kind": "rollout", "tool_names": []}, "GRPO")
    for value in [float("nan"), float("inf"), True]:
        with pytest.raises(ValueError):
            normalize_row(rollout(score=value), "GRPO")


def source(tmp_path, rows):
    path = tmp_path / "raw.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return [{"name": "raw", "path": str(path), "kind": "rollout", "expected_records": len(rows)}]


def test_atomic_bundle_verifies_hashes_and_aggregates_and_never_overwrites(tmp_path):
    specs = source(tmp_path, [rollout(trial=0), rollout(score=1, trial=1)])
    out = tmp_path / "report"
    assert build_bundle(specs, out)["records"] == 2
    assert main(["--verify-dir", str(out)]) == 0
    original = (out / "records.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        build_bundle(specs, out)
    assert (out / "records.jsonl").read_bytes() == original
    summary = json.loads((out / "summary.json").read_text())
    summary["sources"]["raw"]["successes"] = 55
    (out / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="hash"):
        verify_bundle(out)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["outputs_sha256"]["summary.json"] = sha256_file(out / "summary.json")
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="disagree"):
        verify_bundle(out)


def test_duplicate_trajectories_and_incomplete_inputs_never_publish(tmp_path):
    specs = source(tmp_path, [rollout(trial=0), rollout(trial=0)])
    with pytest.raises(ValueError, match="Duplicate"):
        build_bundle(specs, tmp_path / "report")
    assert not (tmp_path / "report").exists()
    specs = source(tmp_path, [rollout(trial=0)])
    specs[0]["expected_records"] = 240
    with pytest.raises(ValueError, match="count"):
        build_bundle(specs, tmp_path / "report")
    with pytest.raises(ValueError, match="uniquely"):
        build_bundle(specs * 2, tmp_path / "report")


def test_cli_manifest_uses_manifest_relative_paths(tmp_path):
    specs = source(tmp_path, [rollout(trial=0)])
    specs[0]["path"] = "raw.jsonl"
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps({"sources": specs}))
    assert main(["--input-manifest", str(manifest), "--out", str(tmp_path / "result")]) == 0


def test_error_sidecar_retained_and_invalidates_rate(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    path.write_text(json.dumps(rollout(score=1, trial=0)) + "\n")
    errors = tmp_path / "errors.jsonl"
    errors.write_text(json.dumps({"task_id": "task", "trial": 1, "seed": 43,
                                 "error": "endpoint unavailable",
                                 "execution_eligibility": execution_eligibility(None, exception=True)}) + "\n")
    out = tmp_path / "result"
    assert build_bundle([{"name": "eval", "path": str(path), "kind": "rollout", "expected_records": 2}], out)["records"] == 2
    summary = json.loads((out / "summary.json").read_text())["sources"]["eval"]
    assert summary["unresolved"] == 1 and summary["success_rate"] is None
    assert summary["badcase_counts"]["infrastructure_error"] == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["sources"][0]["input_files"]) == 2


def test_different_simulation_ids_cannot_hide_duplicate_trials(tmp_path):
    rows = [rollout(trial=0, seed=42, simulation={"id": "a"}),
            rollout(trial=0, seed=43, simulation={"id": "b"})]
    with pytest.raises(ValueError, match="Duplicate"):
        build_bundle(source(tmp_path, rows), tmp_path / "result")


def test_pinned_source_hash_drift_prevents_publication(tmp_path):
    specs = source(tmp_path, [rollout(trial=0)])
    specs[0]["sha256"] = "a" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        build_bundle(specs, tmp_path / "result")
    assert not (tmp_path / "result").exists()
