#!/usr/bin/env bash
# External simulator only: keep the policy's training environment unchanged.
set -euo pipefail
TAU3_ROOT="${TAU3_ROOT:-/root/autodl-fs/tau3_grpo_fix}"
SIM_PYTHON="${TAU3_SIM_PYTHON:-${TAU3_ROOT}/runtime/venvs/qwen38-sim/bin/python}"
MODEL="${TAU3_USER_MODEL:-${TAU3_ROOT}/model_store/Qwen3.8-27B-AWQ-INT4}"
export CUDA_VISIBLE_DEVICES="${TAU3_USER_CUDA_DEVICES:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export TAU3_SIM_CACHE_ROOT="${TAU3_SIM_CACHE_ROOT:-${TAU3_SCRATCH_ROOT:-/root/autodl-tmp/tau3}/cache/qwen38-sim}"
if [[ -n "${TAU3_SIM_CACHE_ROOT:-}" ]]; then
  export XDG_CACHE_HOME="${TAU3_SIM_CACHE_ROOT}"
  export VLLM_CACHE_ROOT="${TAU3_SIM_CACHE_ROOT}/vllm"
  export TRITON_CACHE_DIR="${TAU3_SIM_CACHE_ROOT}/triton"
  export TORCHINDUCTOR_CACHE_DIR="${TAU3_SIM_CACHE_ROOT}/inductor"
  export HF_HOME="${TAU3_SIM_CACHE_ROOT}/huggingface"
fi
ARGS=("${SIM_PYTHON}" -m vllm.entrypoints.cli.main serve "${MODEL}"
  --served-model-name "${TAU3_USER_SERVED_MODEL_NAME:-Qwen/Qwen3.8-27B-AWQ-INT4}"
  --host "${TAU3_USER_HOST:-127.0.0.1}" --port "${TAU3_USER_PORT:-8100}"
  --tensor-parallel-size "${TAU3_USER_TP:-1}" --dtype bfloat16 --quantization compressed-tensors
  --language-model-only
  --default-chat-template-kwargs '{"enable_thinking":false}'
  --enable-prefix-caching --max-model-len "${TAU3_USER_MAX_MODEL_LEN:-16384}"
  --max-num-seqs "${TAU3_USER_MAX_NUM_SEQS:-16}"
  --gpu-memory-utilization "${TAU3_USER_GPU_MEMORY_UTILIZATION:-0.65}")
if [[ "${TAU3_USER_ENFORCE_EAGER:-0}" == "1" ]]; then
  ARGS+=(--enforce-eager)
fi
if [[ "${TAU3_SIMULATOR_COLOCATED_SLEEP:-0}" == "1" ]]; then
  export VLLM_SERVER_DEV_MODE=1
  ARGS+=(--enable-sleep-mode)
fi
ARGS+=("$@")
if [[ "${TAU3_DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${ARGS[@]}"
  printf '\n'
  exit 0
fi
exec "${ARGS[@]}"
