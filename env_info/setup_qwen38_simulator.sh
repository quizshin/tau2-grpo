#!/usr/bin/env bash
# Independent simulator overlay; never upgrades Torch/vLLM in the RL environment.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
TAU3_ROOT="${TAU3_ROOT:-/root/autodl-fs/tau3_grpo_fix}"
BASE_VENV="${TAU3_BASE_VENV:-${TAU3_ROOT}/runtime/venvs/qwen35}"
SIM_VENV="${TAU3_SIM_VENV:-${TAU3_ROOT}/runtime/venvs/qwen38-sim}"
UV="${TAU3_UV:-${TAU3_ROOT}/runtime/bin/uv}"
if [[ "${BASE_VENV}" == "${SIM_VENV}" ]]; then
  echo 'Simulator overlay must be separate from the training environment.' >&2
  exit 2
fi
if [[ ! -x "${SIM_VENV}/bin/python" ]]; then
  "${UV}" venv --python "${BASE_VENV}/bin/python" "${SIM_VENV}"
fi
SITE="${SIM_VENV}/lib/python3.12/site-packages"
# Plain .pth entries expose libraries, not another environment's startup hooks.
# Overlay packages are searched before these read-only shared dependencies.
mkdir -p "${SITE}"
SHARED_PATHS=("${BASE_VENV}/lib/python3.12/site-packages")
if [[ -n "${TAU3_RUNTIME_CACHE:-}" ]]; then
  test -d "${TAU3_RUNTIME_CACHE}/lib/python3.12/site-packages"
  SHARED_PATHS=("${TAU3_RUNTIME_CACHE}/lib/python3.12/site-packages" "${SHARED_PATHS[@]}")
fi
printf '%s\n' "${SHARED_PATHS[@]}" > "${SITE}/tau3_shared_runtime.pth"
"${UV}" pip install --python "${SIM_VENV}/bin/python" --no-deps \
  --index-url "${TAU3_PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple}" \
  'transformers==5.8.0'
"${SIM_VENV}/bin/python" - <<'PY'
import sys
from importlib.metadata import distribution, requires, version
from packaging.requirements import Requirement
from packaging.version import Version
for spec in requires('transformers'):
    req = Requirement(spec)
    if req.marker and not req.marker.evaluate({'extra': ''}):
        continue
    assert Version(version(req.name)) in req.specifier, spec
for name in ('transformers', 'torch', 'vllm', 'compressed-tensors'):
    print(name, version(name), distribution(name).locate_file(''))
assert version('transformers') == '5.8.0'
assert version('vllm') == '0.20.0', 'Revalidate the simulator on other vLLM versions'
print('simulator prefix:', sys.prefix)
PY
