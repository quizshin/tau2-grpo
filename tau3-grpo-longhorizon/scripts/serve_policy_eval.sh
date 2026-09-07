#!/usr/bin/env bash
# Serve one frozen/merged policy checkpoint for selection or official evaluation.
set -euo pipefail

CHECKPOINT="${1:?usage: serve_policy_eval.sh <merged-hf-checkpoint>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${CODE_ROOT}/tau2-bench/src:${PYTHONPATH:-}"
SERVED_NAME="${TAU3_POLICY_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
HOST="${TAU3_POLICY_HOST:-127.0.0.1}"
PORT="${TAU3_POLICY_PORT:-8000}"
TP="${TAU3_POLICY_TP:-2}"
export CUDA_VISIBLE_DEVICES="${TAU3_POLICY_CUDA_DEVICES:-0,1}"
ATTESTATION="${TAU3_POLICY_ATTESTATION_PATH:-${PROJECT_ROOT}/results/policy_service_attestation.json}"
BASE_URL="http://${HOST}:${PORT}/v1"

# exec preserves this shell's PID, so evaluation can reject a stale attestation
# after the serving process exits. The content hash binds the endpoint launch to
# the exact merged checkpoint rather than only its display name.
python -m tau3_grpo.cli.attest_service \
  --checkpoint "${CHECKPOINT}" \
  --served-model-name "${SERVED_NAME}" \
  --base-url "${BASE_URL}" \
  --pid "$$" \
  --output "${ATTESTATION}"

exec vllm serve "${CHECKPOINT}" \
  --served-model-name "${SERVED_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --dtype bfloat16 \
  --max-model-len "${TAU3_POLICY_MAX_MODEL_LEN:-24576}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
