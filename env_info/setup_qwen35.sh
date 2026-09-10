#!/usr/bin/env bash
# Fresh environment: do not inherit the old Torch/vLLM installation.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${TAU3_VENV_DIR:-${ROOT_DIR}/.venv-a800-qwen35}"
"${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
if [[ "$(uname -s)" != "Linux" ]]; then
  echo 'The a800-qwen35 environment is for the remote Linux NVIDIA host.' >&2
  exit 2
fi
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install \
  --constraint "${ROOT_DIR}/env_info/qwen35-constraints.txt" \
  -e "${ROOT_DIR}/tau2-bench" \
  -e "${ROOT_DIR}/verl[qwen35]" \
  -e "${ROOT_DIR}[dev,data,stats,qwen35,tracking]"
"${VENV_DIR}/bin/python" -m pip check
SITE_PACKAGES="$("${VENV_DIR}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n%s\n%s\n' "${ROOT_DIR}" \
  "${ROOT_DIR}/tau2-bench/src" "${ROOT_DIR}/verl" \
  > "${SITE_PACKAGES}/tau3_grpo_local_sources.pth"
echo "Qwen3.5 dependencies installed. GPU validation is still required."
echo "activate with: source ${VENV_DIR}/bin/activate"
