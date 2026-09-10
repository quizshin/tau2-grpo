#!/usr/bin/env bash
# Source from Qwen3.5 entrypoints. No changes to legacy defaults or shared YAML.
QWEN35_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODE_ROOT="${PROJECT_ROOT}"
source "${CODE_ROOT}/scripts/lib/paths.sh"
export PYTHONPATH="${PROJECT_ROOT}:${CODE_ROOT}/tau2-bench/src:${CODE_ROOT}/verl:${PYTHONPATH:-}"

QWEN35_SIZE="${QWEN35_SIZE:-0.8B}"
case "${QWEN35_SIZE}" in
  0.8B|2B|4B|9B) ;;
  *) echo 'QWEN35_SIZE must be 0.8B, 2B, 4B or 9B (dense models)' >&2; exit 2 ;;
esac
QWEN35_MODEL_PATH="${QWEN35_MODEL_PATH:-${TAU3_MODEL_ROOT}/Qwen3.5-${QWEN35_SIZE}}"
QWEN35_SIZE_SLUG="$(printf '%s' "${QWEN35_SIZE}" | tr '[:upper:]' '[:lower:]')"
QWEN35_RUN_ROOT="${QWEN35_RUN_ROOT:-${TAU3_RUN_ROOT}/qwen35_${QWEN35_SIZE_SLUG}}"
# Keep the legacy override name for callers selecting a LoRA checkpoint.
QWEN35_SFT_METHOD="${QWEN35_SFT_METHOD:-full}"
case "${QWEN35_SFT_METHOD}" in
  full|lora) ;;
  *) echo 'QWEN35_SFT_METHOD must be full or lora' >&2; exit 2 ;;
esac
QWEN35_SFT_CHECKPOINT="${QWEN35_SFT_CHECKPOINT:-${QWEN35_SFT_ADAPTER:-${QWEN35_RUN_ROOT}/sft_${QWEN35_SFT_METHOD}_seed42}}"
QWEN35_SFT_ADAPTER="${QWEN35_SFT_CHECKPOINT}"
QWEN35_SFT_MERGED="${QWEN35_SFT_MERGED:-${QWEN35_RUN_ROOT}/sft_merged_seed42}"
# Use the pinned local simulator snapshot; share its API name across all clients.
# Resolve the name before filling the default path so explicit model overrides
# retain their own name unless the caller supplies a served-model alias.
export TAU3_USER_SERVED_MODEL_NAME="${TAU3_USER_SERVED_MODEL_NAME:-${TAU3_USER_MODEL:-Qwen/Qwen3.5-4B}}"
export TAU3_USER_MODEL="${TAU3_USER_MODEL:-${TAU3_MODEL_ROOT}/Qwen3.5-4B}"
export TAU3_GRPO_TEXT_ONLY=1
