#!/usr/bin/env bash
# One-A800 LoRA SFT warm-start. GPU training only; never run on the Mac.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CODE_ROOT="${PROJECT_ROOT}"
source "${CODE_ROOT}/scripts/lib/paths.sh"

export PYTHONPATH="${PROJECT_ROOT}:${CODE_ROOT}/tau2-bench/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${SFT_GPU:-0}"

MODEL_ARGS=()
if [[ -n "${SFT_MODEL_NAME_OR_PATH:-}" ]]; then
  MODEL_ARGS+=(--model-name-or-path "${SFT_MODEL_NAME_OR_PATH}")
fi

# The Python entrypoint checks the selected model family's runtime contract.
exec python -m tau3_grpo.training.sft.train \
  --config "${SFT_CONFIG:-${PROJECT_ROOT}/configs/train/sft/qwen25_lora.yaml}" \
  "${MODEL_ARGS[@]}" \
  "$@"
