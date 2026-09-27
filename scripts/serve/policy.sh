#!/usr/bin/env bash
# Serve one frozen/merged policy checkpoint for selection or official evaluation.
set -euo pipefail

CHECKPOINT="${1:?usage: serve_policy_eval.sh <merged-hf-checkpoint>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODE_ROOT="${PROJECT_ROOT}"
source "${CODE_ROOT}/scripts/lib/paths.sh"
export PYTHONPATH="${PROJECT_ROOT}:${CODE_ROOT}/tau2-bench/src:${PYTHONPATH:-}"
SERVED_NAME="${TAU3_POLICY_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
HOST="${TAU3_POLICY_HOST:-127.0.0.1}"
PORT="${TAU3_POLICY_PORT:-8000}"
TP="${TAU3_POLICY_TP:-2}"
export CUDA_VISIBLE_DEVICES="${TAU3_POLICY_CUDA_DEVICES:-0,1}"
ATTESTATION="${TAU3_POLICY_ATTESTATION_PATH:-${TAU3_RUN_ROOT}/policy_service_attestation.json}"
BASE_URL="http://${HOST}:${PORT}/v1"

# Select from the checkpoint config, not its advertised API alias.
FAMILY="$(python -c 'import sys; from tau3_grpo.models.compat import model_family; print(model_family(sys.argv[1]))' "${CHECKPOINT}")"
TOOL_PARSER=hermes
MODEL_ARGS=()
if [[ "${FAMILY}" == "qwen35" ]]; then
  TOOL_PARSER=qwen3_coder
  MODEL_ARGS+=(--language-model-only --reasoning-parser qwen3
    --default-chat-template-kwargs '{"enable_thinking":false}')
fi

if [[ "${TAU3_EVAL_HARNESS_PROTOCOL:-}" == "tau3_eval_token_v4" ]]; then
  MODEL_ARGS+=(--generation-config vllm --logprobs-mode processed_logprobs)
fi

# exec preserves this shell's PID, so evaluation can reject a stale attestation
# after the serving process exits. The content hash binds the endpoint launch to
# the exact merged checkpoint rather than only its display name.
python -m tau3_grpo.evaluation.attest_service \
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
  --tool-call-parser "${TOOL_PARSER}" \
  "${MODEL_ARGS[@]}"
