#!/usr/bin/env bash
# Run after training finishes and before changing or releasing the node.
set -euo pipefail
ROOT="${TAU3_FS_ROOT:-/root/autodl-fs/tau3_grpo_fix}"
SCRATCH="${TAU3_SCRATCH_ROOT:-/root/autodl-tmp/tau3}"
mkdir -p "${ROOT}/results"
rsync -a --checksum "${SCRATCH}/runs/" "${ROOT}/results/"
# No --delete: persistent-only experiment records must survive.
