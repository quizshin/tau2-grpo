#!/usr/bin/env bash
# Binary-only installation on A800; never compiles or changes the base venv.
set -euo pipefail
DIAG_ROOT=/root/autodl-fs/tau3-core/code/results/legacy/gdn-20260912
DIAG_VENV=/root/autodl-fs/tau3-core/environment/venvs/qwen35
DIAG_WHEEL=flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl
/root/miniconda3/bin/python - <<'PY'
import json,hashlib
from pathlib import Path
root=Path('/root/autodl-fs/tau3-core/code/results/legacy/gdn-20260912')
assert json.loads((root/'rl64-ieee-replay/summary.json').read_text())['strict_numerical_gate_passed']
source=root/'flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='173b0e1a5d6a0becb4ce11b755605c4b20289a6d17816a00cfd7a69078c3e612'
PY
mkdir -p "$DIAG_ROOT/fa2-overlay"
"$DIAG_VENV/bin/python" -m pip install --only-binary=:all: --no-deps --no-index --no-cache-dir \
    --target "$DIAG_ROOT/fa2-overlay" "$DIAG_ROOT/$DIAG_WHEEL"
