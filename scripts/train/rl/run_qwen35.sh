#!/usr/bin/env bash
# Two-A800 engineering profile: policy on GPU 0, external simulator on GPU 1.
# Small batch/update defaults are diagnostics, not the formal 8 x 8 budget.
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
TAU3_DEFAULTS="$(python -m tau3_grpo.training.rl.runtime_defaults qwen35 --size "${QWEN35_SIZE}")"
eval "${TAU3_DEFAULTS}"
unset TAU3_DEFAULTS
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

exec python -m tau3_grpo.training.rl.runtime_defaults exec-qwen35 -- "${ARM}" "${SEED}" "$@"
