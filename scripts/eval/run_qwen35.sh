#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../lib/qwen35.sh"
TARGET="${1:?usage: run_eval.sh selection|tau3-final <merged-checkpoint>}"
CHECKPOINT="${2:?provide the exact merged checkpoint served by vLLM}"
case "${TARGET}" in
  selection) TARGET=selection ;;
  tau3-final|tau3_final) TARGET=tau3-final ;;
  *) echo 'target must be selection or tau3-final' >&2; exit 2 ;;
esac
shift 2
export TAU3_POLICY_ATTESTATION_PATH="${TAU3_POLICY_ATTESTATION_PATH:-${QWEN35_RUN_ROOT}/policy_service_attestation.json}"
exec python -m tau3_grpo.evaluation.run \
  --harness-protocol "${TAU3_EVAL_HARNESS_PROTOCOL:-tau3_eval_legacy_v1}" \
  --target "${TARGET}" --checkpoint "${CHECKPOINT}" \
  --results-dir "${QWEN35_RUN_ROOT}" \
  --policy-model "${TAU3_POLICY_MODEL:-Qwen/Qwen3.5-${QWEN35_SIZE}}" \
  --policy-base-url "${TAU3_POLICY_BASE_URL:-http://127.0.0.1:8000/v1}" \
  --user-model "${TAU3_USER_SERVED_MODEL_NAME}" \
  --user-base-url "${TAU3_USER_BASE_URL:-http://127.0.0.1:8100/v1}" \
  --trials "${EVAL_TRIALS:-4}" --max-concurrency "${MAX_CONCURRENCY:-4}" \
  "$@"
