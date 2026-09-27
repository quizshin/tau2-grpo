import hashlib
import json

import pytest

from tau3_grpo.experiments.review import main, read_receipt, render


def write(tmp_path, data):
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(data))
    return path


def evaluation():
    return dict(status="incomplete", planned_trajectories=240, completed_trajectories=239,
                failed_trajectories=1, missing_trajectories=0, metrics_valid=False,
                metrics=None)


def test_partial_budget_stop_preserves_raw_failure_and_does_not_infer_benefit(tmp_path):
    path = write(tmp_path, dict(schema="tau3_architecture_acceptance_summary_v1",
                               status="partial_budget_boundary_stop", raw_controller_status="failed",
                               parameter_updates=2, phase_c="not run", api_key="secret"))
    receipt = read_receipt("gpu_acceptance", path)
    experiment, errors = render("acceptance", "s42", [receipt])
    assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for text in (experiment, errors):
        assert "partial_budget_boundary_stop" in text and '"raw_controller_status": "failed"' in text
        assert "not run" in text and "secret" not in text
        assert "不推断算法提升" in text


def test_incomplete_evaluation_keeps_denominator_and_withholds_metrics(tmp_path):
    receipt = read_receipt("evaluation", write(tmp_path, evaluation()))
    assert receipt["attention_required"]
    assert receipt["reported_facts"]["planned_trajectories"] == 240
    assert receipt["reported_facts"]["metrics"] is None


@pytest.mark.parametrize("changes", [
    {"metrics": {"pass@1": 0.5}}, {"metrics_valid": True}, {"missing_trajectories": 2},
    {"completed_trajectories": -1}, {"metrics_valid": "false"},
])
def test_contradictory_evaluation_receipts_are_rejected(tmp_path, changes):
    data = evaluation()
    data.update(changes)
    with pytest.raises(ValueError):
        read_receipt("evaluation", write(tmp_path, data))


def test_completed_evaluation_is_not_promoted_to_training_completion(tmp_path):
    data = evaluation()
    data.update(status="complete", completed_trajectories=240, failed_trajectories=0,
                metrics_valid=True, metrics={"pass@1": 0.5, "pass^4": 0.25})
    receipt = read_receipt("evaluation", write(tmp_path, data))
    assert receipt["reported_status"] == "complete"
    assert receipt["kind"] == "evaluation"
    assert not receipt["attention_required"]


def test_drafts_never_overwrite_indexes_or_previous_human_annotations(tmp_path):
    source = write(tmp_path, evaluation())
    index = tmp_path / "EXPERIMENTS.md"
    index.write_text("human conclusion")
    output = tmp_path / "drafts"
    args = ["--receipt", f"evaluation={source}", "--experiment-id", "e2", "--run-id", "s42",
            "--output-dir", str(output)]
    assert main(args) == 0
    draft = output / "experiment.draft.md"
    draft.write_text("human annotation")
    with pytest.raises(FileExistsError):
        main(args)
    assert index.read_text() == "human conclusion"
    assert draft.read_text() == "human annotation"


def test_unrecognized_receipt_does_not_get_an_invented_status(tmp_path):
    with pytest.raises(ValueError):
        read_receipt("gpu_acceptance", write(tmp_path, {"status": "passed"}))


def test_paused_formal_run_does_not_claim_target_completion(tmp_path):
    path = write(tmp_path, dict(status="paused", pid=123, target=20, session="session-1",
                               updated_at=1, completed_step=10, cloud_steps_verified=10,
                               paused=True, error="do not include arbitrary log text"))
    receipt = read_receipt("formal_controller", path)
    assert receipt["attention_required"]
    assert receipt["reported_facts"]["completed_step"] == 10
    assert receipt["reported_facts"]["target"] == 20
    assert "error" not in receipt["reported_facts"]
