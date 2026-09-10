#!/usr/bin/env bash
# Defaults to the two small models; large simulators stay on persistent storage.
set -euo pipefail
ROOT="${TAU3_FS_ROOT:-/root/autodl-fs/tau3_grpo_fix}"
exec "${ROOT}/runtime/venvs/qwen35/bin/python" -S -B "${ROOT}/code/env_info/autodl/restore_models.py" "$@"
