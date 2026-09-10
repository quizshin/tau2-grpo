#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/qwen35.sh"
QWEN35_SFT_CONFIG="${PROJECT_ROOT}/configs/train/sft/qwen35_full.yaml"
if [[ "${QWEN35_SFT_METHOD}" == "lora" ]]; then
  QWEN35_SFT_CONFIG="${PROJECT_ROOT}/configs/train/sft/qwen35_lora.yaml"
fi
export SFT_CONFIG="${SFT_CONFIG:-${QWEN35_SFT_CONFIG}}"
export SFT_MODEL_NAME_OR_PATH="${QWEN35_MODEL_PATH}"
exec bash "${PROJECT_ROOT}/scripts/train/sft/run_base.sh" --output-dir "${QWEN35_SFT_ADAPTER}" "$@"
