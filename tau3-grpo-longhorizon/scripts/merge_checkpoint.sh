#!/usr/bin/env bash
# Convert a veRL FSDP actor checkpoint into the Hugging Face directory vLLM serves.
set -euo pipefail

ACTOR_DIR="${1:?usage: merge_checkpoint.sh <global_step_N/actor> [target-dir]}"
TARGET_DIR="${2:-${ACTOR_DIR%/}/merged_hf}"

if [[ ! -d "${ACTOR_DIR}" ]]; then
  echo "error: actor checkpoint directory not found: ${ACTOR_DIR}" >&2
  exit 2
fi

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${ACTOR_DIR}" \
  --target_dir "${TARGET_DIR}"

echo "merged checkpoint: ${TARGET_DIR}"
