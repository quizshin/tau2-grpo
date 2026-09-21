import json

import pytest

from tau3_grpo.evaluation.compare import compare_evaluations, main, render_markdown
from tau3_grpo.evaluation.diagnostics import trajectory_diagnostics
from tau3_grpo.evaluation.metrics import paired_bootstrap


def artifact(path, successes=(0, 0), *, incomplete=False):
    path.mkdir()
    jobs = [{"task_id": task, "trial": trial, "seed": 42 + trial}
            for task in ("a", "b") for trial in range(4)]
    metadata = {
        "schema_version": 1, "benchmark_revision": "pinned",
        "spec": {"target": "selection", "trials": 4, "seed": 42, "max_steps": 30,
                 "max_errors": 10, "max_concurrency": 4, "ks": [1, 2, 4], "include_pass_hat": True},
        "planned": jobs, "endpoints": {"policy": {"model": path.name, "temperature": .4},
                                       "user": {"model": "simulator", "temperature": 1.0}},
        "provenance": {"tool_protocol": "sequential", "agent_system_prompt_sha256": "prompt",
                       "benchmark_policy_modified": True},
    }
    (path / "run.json").write_text(json.dumps(metadata))
    rows = [{**job, "reward": float(job["trial"] < successes["ab".index(job["task_id"])]),
             "termination_reason": "user_stop"} for job in jobs]
    errors = []
    if incomplete:
        missing = rows.pop()
        errors.append({key: missing[key] for key in ("task_id", "trial", "seed")})
    (path / "trajectories.jsonl").write_text("\n".join(map(json.dumps, rows)))
    (path / "errors.jsonl").write_text("\n".join(map(json.dumps, errors)))
    return path


def change(path, transform):
    data = json.loads((path / "run.json").read_text())
    transform(data)
    (path / "run.json").write_text(json.dumps(data))


def test_exact_paired_deltas_zero_baseline_and_provenance(tmp_path):
    base = artifact(tmp_path / "base")
    treat = artifact(tmp_path / "treat", (2, 4))
    report = compare_evaluations(base, treat, resamples=200)
    assert report["comparable"]
    assert report["differences"]["pass@1"]["difference_pp"] == 75
    assert report["differences"]["pass@1"]["relative_change_percent"] is None
    assert report["differences"]["pass^4"]["treatment_mean"] == .5
    assert report["protocol"]["assurance"] == "recorded_protocol_only"
    assert "provenance.task_db_sha256" in report["protocol"]["missing_extended_or_identity_evidence"]
    assert report == compare_evaluations(base, treat, resamples=200)
    assert "NA" in render_markdown(report)


def test_incomplete_evaluation_cannot_improve_by_dropping_trial(tmp_path):
    base = artifact(tmp_path / "base", (2, 2))
    treat = artifact(tmp_path / "treat", (4, 4), incomplete=True)
    report = compare_evaluations(base, treat)
    assert report["status"] == "incomplete_evaluation"
    assert report["differences"] is None
    assert report["runs"]["treatment"]["score"]["metrics"] is None
    assert not report["comparable"]


@pytest.mark.parametrize("field", ["seed", "max_steps", "target", "max_concurrency"])
def test_protocol_mismatch_is_rejected(tmp_path, field):
    base, treat = artifact(tmp_path / "a"), artifact(tmp_path / "b")
    change(treat, lambda d: d["spec"].update({field: "different"}))
    assert compare_evaluations(base, treat)["status"] == "incompatible_protocol"


def test_missing_core_evidence_is_not_protocol_match(tmp_path):
    base, treat = artifact(tmp_path / "a"), artifact(tmp_path / "b")
    for p in (base, treat):
        change(p, lambda d: d["provenance"].pop("tool_protocol"))
    assert not compare_evaluations(base, treat)["comparable"]


def test_same_task_ids_with_different_data_hash_are_rejected(tmp_path):
    base, treat = artifact(tmp_path / "a"), artifact(tmp_path / "b")
    for p in (base, treat):
        change(p, lambda d: d["provenance"].update(task_db_sha256=p.name))
    assert not compare_evaluations(base, treat)["comparable"]


def test_no_task_intersection_or_cached_score_shortcut(tmp_path):
    base, treat = artifact(tmp_path / "a"), artifact(tmp_path / "b")
    for filename in ("run.json", "trajectories.jsonl"):
        p = treat / filename
        p.write_text(p.read_text().replace('"task_id": "b"', '"task_id": "c"'))
    (treat / "summary.json").write_text('{"metrics_valid": true}')
    report = compare_evaluations(base, treat)
    assert report["differences"] is None
    assert any(m["field"] == "planned" for m in report["protocol"]["mismatches"])


def test_seed_mismatch_duplicate_and_nonfinite_reward_fail_closed(tmp_path):
    base, treat = artifact(tmp_path / "a"), artifact(tmp_path / "b")
    path = treat / "trajectories.jsonl"
    original = path.read_text()
    for invalid in (original.replace('"seed": 42', '"seed": 41', 1),
                    original + "\n" + original.splitlines()[0],
                    original.replace('"reward": 0.0', '"reward": NaN', 1)):
        path.write_text(invalid)
        with pytest.raises(ValueError):
            compare_evaluations(base, treat)


def test_diagnostics_missing_usage_user_tools_and_multi_calls():
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}],
         "usage": {"prompt_tokens": 100, "completion_tokens": 8}},
        {"role": "tool", "id": "a", "requestor": "assistant", "error": False},
        {"role": "tool", "id": "b", "requestor": "assistant", "error": True},
        {"role": "tool", "id": "u", "requestor": "user", "error": True},
        {"role": "assistant", "content": "done"},
    ]
    d = trajectory_diagnostics([{"termination_reason": "max_steps", "simulation": {
        "messages": messages, "duration": 3}}, {"reward": 1}])
    assert d["assistant_tool_calls_per_trajectory"]["coverage"] == .5
    assert d["assistant_tools"]["requested"] == 2
    assert d["assistant_tools"]["error_rate_observed"] == .5
    assert d["assistant_tools"]["response_coverage"] == 1
    usage = d["usage_by_message"]["assistant"]["prompt_tokens"]
    assert usage["sum_observed"] == 100 and usage["coverage"] == .5
    assert d["usage_by_message"]["user"]["prompt_tokens"]["sum_observed"] is None
    assert d["termination_reasons"] == {"max_steps": 1, "unknown": 1}
    assert trajectory_diagnostics([{}])["assistant_tools"]["errors"] is None


def test_compare_cli_refuses_overwrite_and_records_invalid(tmp_path):
    base = artifact(tmp_path / "base")
    treat = artifact(tmp_path / "treat", incomplete=True)
    output = tmp_path / "report"
    args = ["--baseline", str(base), "--treatment", str(treat), "--output-dir", str(output)]
    assert main(args) == 1
    original = (output / "comparison.json").read_bytes()
    assert main(args) == 2
    assert (output / "comparison.json").read_bytes() == original


@pytest.mark.parametrize("kwargs", [{"resamples": 0}, {"resamples": -1}, {"confidence": 1}])
def test_bootstrap_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        paired_bootstrap([1], [0], **kwargs)


def test_bootstrap_rejects_nonfinite_scores():
    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap([float("nan")], [0])


@pytest.mark.parametrize("baseline_version", [None, "tau3_eval_legacy_v1", "tau3_eval_train_control_v2"])
def test_harness_version_blocks_unmatched_comparison(tmp_path, baseline_version):
    base, treat = artifact(tmp_path / "a"), artifact(tmp_path / "b")
    if baseline_version:
        change(base, lambda d: d["provenance"].update(harness_protocol={"version": baseline_version}))
    change(treat, lambda d: d["provenance"].update(harness_protocol={"version": "tau3_eval_train_control_v2"}))
    report = compare_evaluations(base, treat)
    assert report["comparable"] == (baseline_version == "tau3_eval_train_control_v2")


def test_token_protocol_requires_matching_tokenizer_identity(tmp_path):
    from tau3_grpo.evaluation.harness import TOKENS_V4, protocol_metadata

    base, treat = artifact(tmp_path / "a"), artifact(tmp_path / "b")
    for path in (base, treat):
        change(path, lambda d: d["provenance"].update(harness_protocol=protocol_metadata(TOKENS_V4)))
    assert not compare_evaluations(base, treat)["comparable"]
    for path in (base, treat):
        change(path, lambda d: d["provenance"].update(tokenizer_files_sha256={"tokenizer.json": "same"}))
    assert compare_evaluations(base, treat)["comparable"]
    change(treat, lambda d: d["provenance"].update(tokenizer_files_sha256={"tokenizer.json": "changed"}))
    assert not compare_evaluations(base, treat)["comparable"]
