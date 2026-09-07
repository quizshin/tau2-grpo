#!/usr/bin/env bash
# One-A800 LoRA SFT warm-start. GPU training only; never run on the Mac.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/src:${CODE_ROOT}/tau2-bench/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${SFT_GPU:-0}"

MODEL_ARGS=()
if [[ -n "${SFT_MODEL_NAME_OR_PATH:-}" ]]; then
  MODEL_ARGS+=(--model-name-or-path "${SFT_MODEL_NAME_OR_PATH}")
fi

python - <<'PY'
import sys
import torch

assert sys.version_info[:2] == (3, 12), sys.version
assert torch.__version__.startswith("2.8"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available()
PY

exec python -m tau3_grpo.cli.sft_train \
  --config "${PROJECT_ROOT}/configs/sft_airline_lora.yaml" \
  "${MODEL_ARGS[@]}" \
  "$@"
