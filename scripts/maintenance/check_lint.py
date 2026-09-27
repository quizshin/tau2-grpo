"""Reject new Ruff diagnostics while reporting explicitly recorded legacy debt."""
from __future__ import annotations

import collections
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def signature(item):
    path = Path(item["filename"])
    relative = str(path.relative_to(ROOT))
    line = path.read_text().splitlines()[item["location"]["row"] - 1].strip()
    return json.dumps([relative, item["code"], item["message"], line], ensure_ascii=False)


def main():
    baseline = json.loads((ROOT / "env_info/lint_debt.json").read_text())
    version = subprocess.check_output([sys.executable, "-m", "ruff", "--version"], text=True).strip()
    if version != baseline["ruff_version"]:
        raise ValueError(f"Use recorded linter {baseline['ruff_version']}; found {version}")
    command = [sys.executable, "-m", "ruff", "check", "tau3_grpo", "tests", "scripts/maintenance",
               "--output-format", "json"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    actual = collections.Counter(signature(item) for item in json.loads(result.stdout))
    allowed = collections.Counter(baseline["diagnostics"])
    introduced = actual - allowed
    print(json.dumps({"remaining_debt": sum(actual.values()), "recorded_debt": sum(allowed.values()),
                      "resolved": sum((allowed - actual).values()), "introduced": dict(introduced)}, indent=2))
    return int(bool(introduced))


if __name__ == "__main__":
    raise SystemExit(main())
