#!/usr/bin/env bash
# One-A800, one-task, one-trajectory veRL integration smoke.
# This is deliberately outside the formal E0-E3 result directories and does
# not change any frozen experiment default.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/lib/paths.sh"

export CUDA_VISIBLE_DEVICES="${SMOKE_CUDA_DEVICE:-0}"
export MODEL_PATH="${MODEL_PATH:-${TAU3_RUN_ROOT}/sft_airline_merged_seed42}"
export RESULTS_DIR="${RESULTS_DIR:-${TAU3_RUN_ROOT}/smoke/e0_seed42}"
export GROUP_SIZE=1
export GROUPS_PER_UPDATE=1
export PPO_MINI_GROUPS=1
export POLICY_BATCH_DIVISOR=1
export POLICY_GPUS=1
export ROLLOUT_TP=1
export MAX_NUM_SEQS=1
export TOTAL_UPDATES=1
export MAX_USER_TURNS="${MAX_USER_TURNS:-3}"
export MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-3}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
export MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-8192}"
export MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-512}"
export TAU3_USER_API_KEY="${TAU3_USER_API_KEY:-EMPTY}"

# The interaction config points at the independent OpenAI-compatible simulator
# on port 8100. Fail before loading actor/reference weights if that service is
# absent. This check can be skipped only by a launcher that performs an
# equivalent readiness gate itself.
SMOKE_USER_SIMULATOR_MODELS_URL="${SMOKE_USER_SIMULATOR_MODELS_URL:-http://127.0.0.1:8100/v1/models}"
if [[ "${SMOKE_SKIP_USER_SIMULATOR_CHECK:-false}" != "true" ]]; then
  if ! curl --fail --silent --show-error --max-time 5 \
    "${SMOKE_USER_SIMULATOR_MODELS_URL}" >/dev/null; then
    echo "error: user simulator is not ready: ${SMOKE_USER_SIMULATOR_MODELS_URL}" >&2
    echo "start scripts/serve/simulator_base.sh before this smoke" >&2
    exit 2
  fi
fi

exec "${PROJECT_ROOT}/scripts/train/rl/run_base.sh" e0 42 \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  data.dataloader_num_workers=0 \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.actor.fsdp_config.model_dtype="${SMOKE_FSDP_MODEL_DTYPE:-bfloat16}" \
  actor_rollout_ref.ref.fsdp_config.model_dtype="${SMOKE_FSDP_MODEL_DTYPE:-bfloat16}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${SMOKE_PARAM_OFFLOAD:-true}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${SMOKE_OPTIMIZER_OFFLOAD:-true}" \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${SMOKE_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.20}" \
  trainer.save_freq=-1 \
  trainer.rollout_data_dir="${RESULTS_DIR}/rollouts" \
  "$@"
