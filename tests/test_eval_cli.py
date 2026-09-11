"""The public evaluator CLI persists metrics and checkpoint identity together."""

import json
from types import SimpleNamespace

import pytest

from tau3_grpo.evaluation import run, runtime


def test_cli_runs_and_persists_the_report_it_prints(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run, "read_manifest", lambda path: [SimpleNamespace(task_id="a")])
    monkeypatch.setattr(run, "assert_service_matches_checkpoint", lambda **kwargs: SimpleNamespace(
        attestation_hash="attested-service", checkpoint_hash="exact-checkpoint-bytes",
    ))
    jobs = [{"task_id": "a", "trial": i, "seed": 42 + i, "task": object(), "db_path": None}
            for i in range(4)]
    monkeypatch.setattr(runtime, "_selection_jobs", lambda *args: jobs)
    monkeypatch.setattr(runtime, "_run_one", lambda **kwargs: SimpleNamespace(
        reward_info=SimpleNamespace(reward=1 if kwargs["seed"] == 42 else 0),
        termination_reason=SimpleNamespace(value="user_stop"),
        model_dump=lambda mode: {"seed": kwargs["seed"]},
    ))
    assert run.main([
        "--target", "selection", "--checkpoint", "/exact/merged", "--output-dir", str(tmp_path),
        "--include-pass-hat", "--ks", "1", "2", "4",
    ]) == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert printed == saved
    assert saved["metrics"]["pass@4"] == 1
    assert saved["metrics"]["pass^4"] == 0
    assert saved["provenance"]["checkpoint"] == "/exact/merged"
    assert saved["provenance"]["checkpoint_hash"] == "exact-checkpoint-bytes"


def test_dry_run_shows_budget_and_does_not_call_service(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run, "read_manifest", lambda path: [SimpleNamespace(task_id="a")])
    monkeypatch.setattr(run, "assert_service_matches_checkpoint", lambda **kwargs: pytest.fail("service check"))
    assert run.main([
        "--target", "selection", "--checkpoint", "/merged", "--output-dir", str(tmp_path / "out"),
        "--dry-run", "--include-pass-hat",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["planned_trajectories"] == 4
    assert payload["metric_ks"] == [1, 2, 4]
    assert payload["include_pass_hat"]
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("flags", [
    ["--ks", "5"], ["--trials", "0"], ["--max-steps", "0"],
    ["--max-concurrency", "0"], ["--policy-temperature", "nan"],
])
def test_invalid_options_fail_before_loading_data(monkeypatch, flags):
    monkeypatch.setattr(run, "read_manifest", lambda path: pytest.fail("data loaded"))
    assert run.main(["--target", "selection", "--checkpoint", "/merged", *flags]) == 2
