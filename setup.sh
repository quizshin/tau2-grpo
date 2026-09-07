#!/usr/bin/env bash
# Milestone D1: reproducible local/A800 environments and real sibling installs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-cpu-test}"

case "${MODE}" in
  cpu-test)
    PYTHON_BIN="${PYTHON_BIN:-python3.12}"
    VENV_DIR="${ROOT_DIR}/.venv-cpu"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/python" -m pip install --upgrade pip
    "${VENV_DIR}/bin/pip" install -e "${ROOT_DIR}/tau2-bench"
    "${VENV_DIR}/bin/pip" install -e "${ROOT_DIR}/tau3-grpo-longhorizon[dev]"
    ;;
  a800)
    # The remote image is expected to provide Torch 2.8 + CUDA 12.8 already.
    PYTHON_BIN="${PYTHON_BIN:-python3.12}"
    VENV_DIR="${ROOT_DIR}/.venv-a800"
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
    "${VENV_DIR}/bin/python" -m pip install --upgrade pip
    "${VENV_DIR}/bin/pip" install -e "${ROOT_DIR}/tau2-bench"
    "${VENV_DIR}/bin/pip" install -e "${ROOT_DIR}/verl[vllm]"
    "${VENV_DIR}/bin/pip" install -e "${ROOT_DIR}/tau3-grpo-longhorizon[dev]"
    ;;
  *)
    echo "usage: $0 {cpu-test|a800}" >&2
    exit 2
    ;;
esac

echo "environment ready: ${VENV_DIR}"
