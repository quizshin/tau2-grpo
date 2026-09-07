#!/usr/bin/env bash
# A800 preflight: imports, pinned CUDA contract, live schemas and local patches.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CODE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/src:${CODE_ROOT}/tau2-bench/src:${CODE_ROOT}/verl:${PYTHONPATH:-}"

python - <<'PY'
import sys
import torch

assert sys.version_info[:2] == (3, 12), sys.version
assert torch.__version__.startswith("2.8"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 1

import tau2  # noqa: F401
import tau3_grpo  # noqa: F401
import verl  # noqa: F401
import vllm
import transformers

assert vllm.__version__ == "0.10.2", vllm.__version__
assert transformers.__version__ == "4.56.1", transformers.__version__

print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__)
print("transformers", transformers.__version__)
print("gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0))
PY

python -m tau3_grpo.cli.verify_patches --check-registration

TMP_CONFIG="$(mktemp)"
trap 'rm -f "${TMP_CONFIG}"' EXIT
python -m tau3_grpo.cli.generate_tool_config --output "${TMP_CONFIG}"
python - "${TMP_CONFIG}" <<'PY'
import sys
import yaml

config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
tools = config["tools"]
names = {tool["tool_schema"]["function"]["name"] for tool in tools}
assert len(tools) == 14, len(tools)
assert "get_flight_status" in names
assert all(tool["config"]["type"] == "native" for tool in tools)
print("live Airline tools", len(tools))
PY

python -m pytest -q "${PROJECT_ROOT}/tests/test_patch_contract.py"
echo "A800 installation preflight passed"
