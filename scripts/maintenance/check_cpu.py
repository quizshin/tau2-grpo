"""Run an explicit CPU suite with offline assets; never enables a GPU."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("core", "benchmark", "verl", "all"), default="core")
    parser.add_argument("--list", action="store_true")
    args, extra = parser.parse_known_args(argv)
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
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    return subprocess.run([sys.executable, "-m", "pytest", "-q", *selected, *extra], cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
