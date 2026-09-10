#!/usr/bin/env bash
# File-storage Python/venvs remain usable without copying the 8GB environment.
set -euo pipefail
ROOT="${TAU3_FS_ROOT:-/root/autodl-fs/tau3_grpo_fix}"
SCRATCH="${TAU3_SCRATCH_ROOT:-/root/autodl-tmp/tau3}"
if [[ $# -gt 0 ]]; then echo 'Usage: bash restore_node.sh (models: 0.8B + 4B)' >&2; exit 2; fi
mkdir -p "${SCRATCH}/runtime/code" "${SCRATCH}/data" "${SCRATCH}/runs" "${SCRATCH}/cache" "${SCRATCH}/tmp"
# Do not overwrite local result/model links or copy runtime assets through them.
rsync -a --exclude='.venv*' --exclude='/data' --exclude='/models' --exclude='/results' --exclude='__pycache__' --exclude='*.pyc' --exclude='/.cache' "${ROOT}/code/" "${SCRATCH}/runtime/code/"
rsync -a "${ROOT}/data/" "${SCRATCH}/data/"
ln -sfn "${SCRATCH}/data" "${SCRATCH}/runtime/code/data"
ln -sfn "${SCRATCH}/models" "${SCRATCH}/runtime/code/models"
ln -sfn "${SCRATCH}/runs" "${SCRATCH}/runtime/code/results"
if [[ ! -x "${SCRATCH}/runtime/qwen35/bin/python" ]]; then
  ln -sfn "${ROOT}/runtime/venvs/qwen35" "${SCRATCH}/runtime/code/.venv-a800-qwen35"
fi
bash "${ROOT}/restore_models.sh"
printf 'Prepared. Run: source %s/activate.sh\n' "${ROOT}"
printf 'SFT artifacts remain at %s/results/sft_compare_20260908; copy only the checkpoint needed for training.\n' "${ROOT}"
