#!/usr/bin/env bash
# Serve the frozen user simulator on the two GPUs reserved by the 6+2 topology.
set -euo pipefail

MODEL="${TAU3_USER_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${TAU3_USER_SERVED_MODEL_NAME:-${MODEL}}"
HOST="${TAU3_USER_HOST:-127.0.0.1}"
PORT="${TAU3_USER_PORT:-8100}"
TP="${TAU3_USER_TP:-2}"
GPU_MEMORY_UTILIZATION="${TAU3_USER_GPU_MEMORY_UTILIZATION:-0.9}"
export CUDA_VISIBLE_DEVICES="${TAU3_USER_CUDA_DEVICES:-6,7}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/lib/paths.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
FAMILY="$(python -c 'import sys; from tau3_grpo.models.compat import model_family; print(model_family(sys.argv[1]))' "${MODEL}")"
MODEL_ARGS=()
if [[ "${FAMILY}" == "qwen35" ]]; then
  MODEL_ARGS+=(--language-model-only --reasoning-parser qwen3
    --default-chat-template-kwargs '{"enable_thinking":false}')
fi

exec vllm serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --dtype bfloat16 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-prefix-caching \
  --max-model-len "${TAU3_USER_MAX_MODEL_LEN:-16384}" \
  "${MODEL_ARGS[@]}"
