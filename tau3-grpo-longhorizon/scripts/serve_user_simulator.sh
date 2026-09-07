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

exec vllm serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --dtype bfloat16 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-prefix-caching \
  --max-model-len "${TAU3_USER_MAX_MODEL_LEN:-16384}"
