#!/usr/bin/env bash
# Launch one arm through veRL's real PPO entrypoint.
#
# Milestone D8–D11. All paths resolve against the flat `code/` root, so
# the script works from anywhere. GPU only: do not run this on a laptop.
#
# Usage:
#   scripts/train/rl/run_base.sh e3 42
#   ARM=e3 SEED=42 TOTAL_UPDATES=60 scripts/train/rl/run_base.sh
#   scripts/maintenance/run_one_trajectory.sh

set -euo pipefail

ARM="${1:-${ARM:-e0}}"
SEED="${2:-${SEED:-42}}"
# ARM and SEED are wrapper arguments, not Hydra overrides. Consume only the
# positionals that are present so the environment-only invocation keeps working.
if (( $# > 0 )); then
  shift
fi
if (( $# > 0 )); then
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CODE_ROOT="${PROJECT_ROOT}"
source "${CODE_ROOT}/scripts/lib/paths.sh"
TRAIN_MANIFEST_DIR="${TRAIN_MANIFEST_DIR:-${TAU3_DATA_ROOT}/manifests}"

TAU2_SRC="${CODE_ROOT}/tau2-bench/src"
VERL_ROOT="${CODE_ROOT}/verl"
PROJECT_SRC="${PROJECT_ROOT}"

for required in "${TAU2_SRC}" "${VERL_ROOT}" "${PROJECT_SRC}"; do
  if [[ ! -d "${required}" ]]; then
    echo "error: expected directory not found: ${required}" >&2
    exit 2
  fi
done

export PYTHONPATH="${PROJECT_SRC}:${TAU2_SRC}:${VERL_ROOT}:${PYTHONPATH:-}"
if [[ -f "${TAU3_ENV_FILE:-${CODE_ROOT}/.env}" && "${TAU3_TRACKING_ENV_LOADED:-0}" != 1 ]]; then
  exec python -m tau3_grpo.tracking.with_env bash "${BASH_SOURCE[0]}" "${ARM}" "${SEED}" "$@"
fi

# Config-backed legacy defaults; assignment failure stops before eval/training.
TAU3_DEFAULTS="$(python -m tau3_grpo.training.rl.runtime_defaults base)"
eval "${TAU3_DEFAULTS}"
unset TAU3_DEFAULTS

VAL_PARQUET="${VAL_PARQUET:-${TAU3_DATA_ROOT}/parquet/airline_selection_seed${DATA_SPLIT_SEED}.parquet}"
TOOL_CONFIG="${TOOL_CONFIG:-${PROJECT_ROOT}/configs/envs/tool_config.yaml}"
INTERACTION_CONFIG="${INTERACTION_CONFIG:-${PROJECT_ROOT}/configs/envs/interaction_config.yaml}"
RESULTS_DIR="${RESULTS_DIR:-${TAU3_RUN_ROOT}/${ARM}_seed${SEED}}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${RESULTS_DIR}/rollouts}"
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-${RESULTS_DIR}/swanlog}"
CUSTOM_TRAIN_PARQUET="${TRAIN_PARQUET:-}"
TRAIN_PARQUET="${CUSTOM_TRAIN_PARQUET:-${RESULTS_DIR}/train_schedule.parquet}"

# Derived paths/batch sizes remain runtime values, not duplicated parameters.
MODEL_PATH="${MODEL_PATH:-${TAU3_RUN_ROOT}/sft_airline_merged_seed42}"
REAL_ROLLOUTS=$((GROUP_SIZE * GROUPS_PER_UPDATE))
POLICY_BATCH_DIVISOR="${POLICY_BATCH_DIVISOR:-$((PPO_MINI_GROUPS * GROUP_SIZE))}"

if [[ ! -d "${MODEL_PATH}" && "${MODEL_PATH}" != */* ]]; then
  echo "error: MODEL_PATH must be a local merged checkpoint or Hugging Face model id" >&2
  exit 2
fi
if [[ "${TAU3_DRY_RUN:-0}" != "1" && "${MODEL_PATH}" == "${PROJECT_ROOT}"/* && ! -d "${MODEL_PATH}" ]]; then
  echo "error: merged SFT checkpoint missing: ${MODEL_PATH}" >&2
  echo "run: scripts/train/sft/run_base.sh && scripts/train/sft/merge.sh" >&2
  echo "or explicitly set MODEL_PATH=<checkpoint-or-model-id> for a smoke test" >&2
  exit 2
fi

if (( POLICY_GPUS % ROLLOUT_TP != 0 )); then
  echo "error: POLICY_GPUS=${POLICY_GPUS} must be divisible by rollout TP=${ROLLOUT_TP}" >&2
  exit 2
fi

if (( POLICY_BATCH_DIVISOR % POLICY_GPUS != 0 )); then
  echo "error: policy batch divisor ${POLICY_BATCH_DIVISOR} must divide ${POLICY_GPUS} GPUs" >&2
  exit 2
fi

TAU3_ARM_DEFAULTS="$(python -m tau3_grpo.training.rl.runtime_defaults arm --arm "${ARM}")"
eval "${TAU3_ARM_DEFAULTS}"
unset TAU3_ARM_DEFAULTS

export TAU3_GRPO_ANCHOR_VERSION="${TAU3_GRPO_ANCHOR_VERSION:-v1}"
case "${TAU3_GRPO_ANCHOR_VERSION}" in
  v1|v2|v3|v4) ;;
  *) echo "error: unsupported anchor version ${TAU3_GRPO_ANCHOR_VERSION}" >&2; exit 2 ;;
esac

ROLLOUT_BACKEND=vllm

echo "arm=${ARM} seed=${SEED} estimator=${ADV_ESTIMATOR} df=${DF_ENABLE} anchors=${ANCHOR_MODE}/${TAU3_GRPO_ANCHOR_VERSION}"
echo "backend=${ROLLOUT_BACKEND} updates=${TOTAL_UPDATES} results=${RESULTS_DIR}"

if [[ "${TAU3_DRY_RUN:-0}" != "1" ]]; then
mkdir -p "${RESULTS_DIR}"
python -m tau3_grpo.algorithms.anchors.protocol \
  --results-dir "${RESULTS_DIR}" --version "${TAU3_GRPO_ANCHOR_VERSION}"

# Materialise and verify the exact non-shuffled task order documented by the
# experiment manifest. A caller may provide TRAIN_PARQUET explicitly only for a
# diagnostic run with its own audited input.
if [[ -z "${CUSTOM_TRAIN_PARQUET}" ]]; then
  python -m tau3_grpo.experiments.prepare \
    --arm "${ARM}" \
    --seed "${SEED}" \
    --data-seed "${DATA_SPLIT_SEED}" \
    --group-size "${GROUP_SIZE}" \
    --groups-per-update "${GROUPS_PER_UPDATE}" \
    --total-updates "${TOTAL_UPDATES}" \
    --anchor-mode "${ANCHOR_MODE}" \
    --manifest-dir "${TRAIN_MANIFEST_DIR}" \
    --output-dir "${RESULTS_DIR}"
elif [[ ! -f "${TRAIN_PARQUET}" ]]; then
  echo "error: custom training parquet missing: ${TRAIN_PARQUET}" >&2
  exit 2
fi

# Fail before burning GPU hours if a local veRL patch went missing.
python -m tau3_grpo.envs.generate_tool_config --output "${TOOL_CONFIG}"
python -m tau3_grpo.integrations.verify_patches --check-registration
fi

# The wrapper registers tau_gigpo and then executes veRL in this same process;
# rollout workers lazily import the anchor hook through the inherited env var.
export TAU3_GRPO_ANCHOR_HOOK="tau3_grpo.integrations.anchor_hook:current_anchor"
export TAU3_GRPO_ANCHOR_MODE="${ANCHOR_MODE}"
export TAU3_GRPO_ANCHOR_SIMILARITY_THRESHOLD="0.9"
export TAU3_GRPO_TRAIN_SEED="${SEED}"
export TAU3_GRPO_MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN}"
export TAU3_GRPO_TELEMETRY_PATH="${RESULTS_DIR}/telemetry.jsonl"
export TAU3_GRPO_ARM="${ARM}"
export TAU3_GRPO_POLICY_BATCH_DIVISOR="${POLICY_BATCH_DIVISOR}"
TRAIN_COMMAND=(python -m tau3_grpo.training.rl.train \
  algorithm.adv_estimator="${ADV_ESTIMATOR}" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.TAU3_GRPO_ANCHOR_VERSION=${TAU3_GRPO_ANCHOR_VERSION}" \
  algorithm.gamma=1.0 \
  algorithm.use_kl_in_reward=false \
  algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
  "+algorithm.dynamic_filter={enable:${DF_ENABLE},mode:fixed_rollout,group_size:${GROUP_SIZE},max_reward:1.0}" \
  "+algorithm.gigpo={omega:1.0,gamma:0.95,fnorm:1.0,min_anchor_group_size:2,similarity_threshold:0.9,anchor_version:${TAU3_GRPO_ANCHOR_VERSION}}" \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  data.train_batch_size="${GROUPS_PER_UPDATE}" \
  data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  data.seed="${SEED}" \
  data.shuffle=false \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.return_raw_chat=true \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}" \
  actor_rollout_ref.actor.data_loader_seed="${SEED}" \
  actor_rollout_ref.actor.fsdp_config.seed="${SEED}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_GROUPS}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.rollout.name="${ROLLOUT_BACKEND}" \
  actor_rollout_ref.rollout.n="${GROUP_SIZE}" \
  actor_rollout_ref.rollout.temperature="${TRAIN_TEMP}" \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LENGTH}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LENGTH}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${UPDATE_WEIGHTS_BUCKET_MEGABYTES}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.multi_turn.tool_execution_mode=sequential \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_USER_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_ASSISTANT_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_TOOL_RESPONSE_CHARS}" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
  actor_rollout_ref.rollout.multi_turn.interaction_config_path="${INTERACTION_CONFIG}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  trainer.n_gpus_per_node="${POLICY_GPUS}" \
  trainer.nnodes=1 \
  trainer.total_training_steps="${TOTAL_UPDATES}" \
  trainer.project_name="${TAU3_SWANLAB_PROJECT:-tau3-grpo-pytrio}" \
  trainer.experiment_name="${SWANLAB_EXPERIMENT_NAME:-tau3-auto}" \
  trainer.default_local_dir="${RESULTS_DIR}" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  "trainer.logger=${TRAINER_LOGGERS:-[console,swanlab]}" \
  trainer.val_before_train=false \
  trainer.save_freq=10 \
  trainer.test_freq=-1 \
  "$@")

if [[ "${TAU3_DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${TRAIN_COMMAND[@]}"
  printf '\n'
else
  exec "${TRAIN_COMMAND[@]}"
fi
