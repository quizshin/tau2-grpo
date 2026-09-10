#!/usr/bin/env bash
# Two-A800 engineering profile: policy on GPU 0, external simulator on GPU 1.
# Small batch/update defaults are diagnostics, not the formal 16 x 8 budget.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/qwen35.sh"
ARM="${1:-e0}"
SEED="${2:-42}"
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi

export CUDA_VISIBLE_DEVICES="${TAU3_POLICY_CUDA_DEVICES:-0}"
export TAU3_POLICY_DISPLAY_MODEL="${TAU3_POLICY_DISPLAY_MODEL:-Qwen/Qwen3.5-${QWEN35_SIZE}}"
export MODEL_PATH="${MODEL_PATH:-${QWEN35_SFT_MERGED}}"
export RESULTS_DIR="${RESULTS_DIR:-${QWEN35_RUN_ROOT}/${ARM}_seed${SEED}}"
export POLICY_GPUS="${POLICY_GPUS:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export GROUP_SIZE="${GROUP_SIZE:-4}"
export GROUPS_PER_UPDATE="${GROUPS_PER_UPDATE:-2}"
export PPO_MINI_GROUPS="${PPO_MINI_GROUPS:-2}"
export TOTAL_UPDATES="${TOTAL_UPDATES:-3}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
export MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-24576}"
# FSDP transfers FP32 state tensors without casting. The 9B embedding/LM-head
# tensors are 3,880 MiB each and must fit individually in the transfer bucket.
QWEN35_TRANSFER_BUCKET_MB=2560
if [[ "${QWEN35_SIZE}" == "9B" ]]; then
  QWEN35_TRANSFER_BUCKET_MB=4096
fi
export UPDATE_WEIGHTS_BUCKET_MEGABYTES="${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-${QWEN35_TRANSFER_BUCKET_MB}}"
export INTERACTION_CONFIG="${RESULTS_DIR}/interaction_config.yaml"

if [[ "${TAU3_DRY_RUN:-0}" != "1" ]]; then
  python -m tau3_grpo.models.check_qwen35 --model "${MODEL_PATH}" --gpu
  python -m tau3_grpo.envs.simulator_config \
    --output "${INTERACTION_CONFIG}" \
    --model "${TAU3_USER_SERVED_MODEL_NAME}" \
    --base-url "${TAU3_USER_BASE_URL:-http://127.0.0.1:8100/v1}" \
    --thinking "${TAU3_USER_THINKING:-auto}" \
    --max-user-turns "${MAX_USER_TURNS:-15}" \
    --max-assistant-turns "${MAX_ASSISTANT_TURNS:-15}"
fi

exec bash "${PROJECT_ROOT}/scripts/train/rl/run_base.sh" "${ARM}" "${SEED}" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.model.use_fused_kernels=false \
  actor_rollout_ref.actor.freeze_vision_tower=true \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
  actor_rollout_ref.actor.fsdp_config.model_dtype=float32 \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.use_orig_params=true \
  actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.enable_prefix_caching=false \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=true \
  +data.apply_chat_template_kwargs.enable_thinking=false \
  trainer.use_legacy_worker_impl=enable \
  trainer.save_freq=1 \
  "$@"
