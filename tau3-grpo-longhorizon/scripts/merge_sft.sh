#!/usr/bin/env bash
# Merge the trained LoRA adapter before veRL/vLLM loads the SFT warm-start.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ADAPTER="${ADAPTER:-${PROJECT_ROOT}/results/sft_airline_lora_seed42}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/results/sft_airline_merged_seed42}"

exec python -m tau3_grpo.cli.merge_sft \
  --base "${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}" \
  --adapter "${ADAPTER}" \
  --output "${OUTPUT}" \
  "$@"
