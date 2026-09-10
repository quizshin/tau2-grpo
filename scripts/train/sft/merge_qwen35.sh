#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/qwen35.sh"
export BASE_MODEL="${QWEN35_MODEL_PATH}"
export ADAPTER="${QWEN35_SFT_ADAPTER}"
export OUTPUT="${QWEN35_SFT_MERGED}"
exec bash "${PROJECT_ROOT}/scripts/train/sft/merge.sh" "$@"
