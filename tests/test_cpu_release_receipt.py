"""Release evidence must reject incomplete runs and identify tested source."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "cpu_release_runner", Path(__file__).resolve().parents[1] / "scripts/maintenance/check_cpu.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "tests/test_example.py").touch()
    (root / "env_info").mkdir()
    (root / "env_info/cpu_test_suites.json").write_text(
        json.dumps({"suites": {"core": ["tests/test_example.py"], "benchmark": [], "verl": []}})
    )
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(
        runner, "source_identity", lambda: {"head": "same", "tracked_sha256": "same"}
    )
    return root


@pytest.mark.parametrize(
    "xml,returncode,expected",
    [
        ('<testsuite><testcase name="ok"/></testsuite>', 0, "passed"),
        (
            '<testsuite><testcase name="missing"><skipped message="asset absent"/></testcase></testsuite>',
            0,
            "incomplete",
        ),
        ('<testsuite><testcase name="broken"><failure/></testcase></testsuite>', 1, "incomplete"),
        ("<testsuite/>", 0, "incomplete"),
    ],
)
def test_strict_receipt_rejects_skip_failure_and_empty(
    fake_repo, tmp_path, monkeypatch, xml, returncode, expected
):
    def run(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        Path(command[-1].split("=", 1)[1]).write_text(xml)
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(runner.subprocess, "run", run)
    report = tmp_path / "report"
    code = runner.main(["--report-dir", str(report), "--require-no-skips"])
    receipt = json.loads((report / "receipt.json").read_text())
    assert receipt["status"] == expected
    assert (code == 0) == (expected == "passed")


def test_changed_source_cannot_receive_pass(fake_repo, tmp_path, monkeypatch):
    identities = iter([{"head": "before"}, {"head": "after"}])
    monkeypatch.setattr(runner, "source_identity", lambda: next(identities))

    def run(command, **kwargs):
        Path(command[-1].split("=", 1)[1]).write_text("<testsuite><testcase/></testsuite>")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    report = tmp_path / "report"
    assert runner.main(["--report-dir", str(report)]) == 1
    assert "Source changed" in json.loads((report / "receipt.json").read_text())["error"]


def test_missing_tokenizer_fails_without_invoking_pytest(fake_repo, tmp_path, monkeypatch):
    (fake_repo / "tests/test_example.py").rename(fake_repo / "tests/test_call_attribution.py")
    (fake_repo / "env_info/cpu_test_suites.json").write_text(
        json.dumps({"suites": {"benchmark": ["tests/test_call_attribution.py"]}})
    )

    def no_run(*args, **kwargs):
        pytest.fail("must not launch tests or download assets")

    monkeypatch.setattr(runner.subprocess, "run", no_run)
    report = tmp_path / "report"
    assert runner.main(["--suite", "benchmark", "--report-dir", str(report)]) == 1
    receipt = json.loads((report / "receipt.json").read_text())
    assert receipt["status"] == "incomplete"
    assert "no download" in receipt["error"]
