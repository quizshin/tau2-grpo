#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../lib/qwen35.sh"
CHECKPOINT="${1:-${QWEN35_SFT_MERGED}}"
export TAU3_POLICY_MODEL="${TAU3_POLICY_MODEL:-Qwen/Qwen3.5-${QWEN35_SIZE}}"
export TAU3_POLICY_TP="${TAU3_POLICY_TP:-1}"
export TAU3_POLICY_CUDA_DEVICES="${TAU3_POLICY_CUDA_DEVICES:-0}"
export TAU3_POLICY_ATTESTATION_PATH="${TAU3_POLICY_ATTESTATION_PATH:-${QWEN35_RUN_ROOT}/policy_service_attestation.json}"
exec bash "${PROJECT_ROOT}/scripts/serve/policy.sh" "${CHECKPOINT}"
