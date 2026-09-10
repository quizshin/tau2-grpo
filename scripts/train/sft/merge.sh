#!/usr/bin/env bash
# Merge the trained LoRA adapter before veRL/vLLM loads the SFT warm-start.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CODE_ROOT="${PROJECT_ROOT}"
source "${CODE_ROOT}/scripts/lib/paths.sh"
export PYTHONPATH="${PROJECT_ROOT}:${CODE_ROOT}/tau2-bench/src:${PYTHONPATH:-}"
ADAPTER="${ADAPTER:-${TAU3_RUN_ROOT}/sft_airline_lora_seed42}"
OUTPUT="${OUTPUT:-${TAU3_RUN_ROOT}/sft_airline_merged_seed42}"

exec python -m tau3_grpo.training.sft.merge \
  --base "${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}" \
  --adapter "${ADAPTER}" \
  --output "${OUTPUT}" \
  "$@"
