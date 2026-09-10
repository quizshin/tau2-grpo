#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../lib/qwen35.sh"
export TAU3_USER_CUDA_DEVICES="${TAU3_USER_CUDA_DEVICES:-1}"
export TAU3_USER_TP="${TAU3_USER_TP:-1}"
exec bash "${PROJECT_ROOT}/scripts/serve/simulator_base.sh" "$@"
