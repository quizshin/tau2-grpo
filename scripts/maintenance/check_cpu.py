"""Run an explicit CPU suite with offline assets; never enables a GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def source_identity():
    """Bind evidence to tracked bytes, including dirty changes, without recording secrets."""
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    digest = hashlib.sha256()
    for name in sorted(x for x in names if x):
        path = ROOT / os.fsdecode(name)
        data = (
            os.readlink(path).encode()
            if path.is_symlink()
            else path.read_bytes()
            if path.is_file()
            else b"<missing>"
        )
        digest.update(name + b"\0" + hashlib.sha256(data).digest())
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tracked_sha256": digest.hexdigest(),
        "status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
    }


def junit_summary(path):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    skipped = [
        {
            "test": case.get("classname", "") + "." + case.get("name", ""),
            "reason": item.get("message", ""),
        }
        for case in cases
        for item in case.findall("skipped")
    ]
    return {
        "tests": len(cases),
        "failures": sum(c.find("failure") is not None for c in cases),
        "errors": sum(c.find("error") is not None for c in cases),
        "skipped": skipped,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("core", "benchmark", "verl", "all"), default="core")
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--report-dir", type=Path, help="New directory for JUnit and identity receipt"
    )
    parser.add_argument(
        "--require-no-skips",
        action="store_true",
        help="Fail release acceptance when any selected test is skipped",
    )
    args, extra = parser.parse_known_args(argv)
    if args.require_no_skips and args.report_dir is None:
        parser.error("--require-no-skips requires --report-dir")
    if args.report_dir and any(x.startswith(("--junitxml", "--junit-xml")) for x in extra):
        parser.error("--report-dir owns the JUnit output; do not override it")
    suites = json.loads((ROOT / "env_info/cpu_test_suites.json").read_text())["suites"]
    files = sorted({name for values in suites.values() for name in values})
    if set(files) != {str(path.relative_to(ROOT)) for path in (ROOT / "tests").glob("test_*.py")}:
        raise ValueError("Update cpu_test_suites.json: new, missing or unclassified test module")
    if sum(map(len, suites.values())) != len(files):
        raise ValueError("A test module belongs to multiple CPU suites")
    selected = files if args.suite == "all" else suites[args.suite]
    if args.list:
        print("\n".join(selected))
        return 0
    env = dict(
        os.environ,
        PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    if args.report_dir is None:
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *selected, *extra], cwd=ROOT, env=env
        ).returncode
    report = args.report_dir.resolve()
    report.mkdir(parents=True, exist_ok=False)
    before = source_identity()
    receipt = {
        "suite": args.suite,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "executable": sys.executable,
        "source_before": before,
        "cuda_visible_devices": "",
        "offline": True,
        "selected_modules": selected,
        "require_no_skips": args.require_no_skips,
        "status": "incomplete",
    }
    code = 1
    try:
        if "tests/test_call_attribution.py" in selected:
            tokenizer = ROOT / "models/Qwen3.5-4B/tokenizer.json"
            if not tokenizer.is_file():
                raise FileNotFoundError("Existing tokenizer.json required; no download performed")
            receipt["tokenizer_sha256"] = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
        junit = report / "junit.xml"
        command = [sys.executable, "-m", "pytest", "-q", *selected, *extra, f"--junitxml={junit}"]
        receipt["command"] = command
        code = subprocess.run(command, cwd=ROOT, env=env).returncode
        receipt["pytest_exit_code"] = code
        summary = junit_summary(junit)
        receipt.update(summary)
        after = source_identity()
        receipt["source_after"] = after
        if before != after:
            raise RuntimeError("Source changed during tests; receipt cannot certify one version")
        if summary["tests"] == 0 or summary["failures"] or summary["errors"]:
            code = code or 1
        if summary["skipped"] and args.require_no_skips:
            code = code or 1
        receipt["status"] = (
            "passed"
            if not code and not summary["skipped"]
            else "passed_with_skips"
            if not code
            else "incomplete"
        )
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        code = code or 1
    finally:
        receipt["exit_code"] = code
        (report / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"CPU receipt: {report / 'receipt.json'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
