#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"
exec python -m tau3_grpo.launch sft "$@"
