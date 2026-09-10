#!/usr/bin/env bash
# End-to-end CPU preparation: manifests then parquet, for both seeds.
#
# Milestone D1–D2, D8. Safe to run on any machine; no GPU, no training.
#
# Usage:
#   scripts/data/prepare.sh
#   DOWNLOAD=1 REQUIRE_DB=1 scripts/data/prepare.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODE_ROOT="${PROJECT_ROOT}"
source "${CODE_ROOT}/scripts/lib/paths.sh"

export PYTHONPATH="${PROJECT_ROOT}:${CODE_ROOT}/tau2-bench/src:${PYTHONPATH:-}"

PREPARE_FLAGS=()
[[ "${DOWNLOAD:-0}" == "1" ]] && PREPARE_FLAGS+=(--download)
[[ "${REQUIRE_DB:-0}" == "1" ]] && PREPARE_FLAGS+=(--require-db)

DATA_SPLIT_SEED="${DATA_SPLIT_SEED:-42}"
SFT_MODEL_NAME_OR_PATH="${SFT_MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B-Instruct}"

echo "=== data split seed ${DATA_SPLIT_SEED}: manifests ==="
python -m tau3_grpo.data.prepare --seed "${DATA_SPLIT_SEED}" "${PREPARE_FLAGS[@]}"

echo "=== data split seed ${DATA_SPLIT_SEED}: parquet ==="
for SPLIT in train selection; do
  python -m tau3_grpo.data.build_parquet --seed "${DATA_SPLIT_SEED}" --split "${SPLIT}"
done

echo "=== data split seed ${DATA_SPLIT_SEED}: 45+5 complete-dialogue SFT ==="
python -m tau3_grpo.data.prepare_sft \
  --seed "${DATA_SPLIT_SEED}" \
  --model-name-or-path "${SFT_MODEL_NAME_OR_PATH}"

echo
echo "done. one frozen split is reused by every independent training seed"
