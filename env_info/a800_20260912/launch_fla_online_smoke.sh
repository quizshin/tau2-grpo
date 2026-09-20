#!/usr/bin/env bash
set -euo pipefail
DIAG_CODE=/root/autodl-fs/tau3-core/code
DIAG_VENV=/root/autodl-fs/tau3-core/environment/venvs/qwen35
export TAU3_ROOT=/root/autodl-fs/tau3-core
export TAU3_DATA_ROOT="$TAU3_ROOT/code/data"
export TAU3_MODEL_ROOT=/root/autodl-fs/tau3-core/code/models
export TAU3_RUN_ROOT="$TAU3_ROOT/code/results/runs"
export TAU3_CACHE_ROOT=/root/autodl-fs/tau3-core/code/.cache/five-a800
export TAU3_SIM_PYTHON="$TAU3_ROOT/environment/venvs/qwen38-sim/bin/python"
export TAU3_FLA_ONLINE_DIR="$TAU3_ROOT/code/results/runs/fla-online-20260912"
export PATH="$DIAG_VENV/bin:$PATH"
export PYTHONPATH="$DIAG_CODE:$DIAG_CODE/verl:$DIAG_CODE/tau2-bench/src:/root/autodl-fs/tau3-core/environment/overlays/fla"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
case "${1:-run}" in
  bridge)
    export CUDA_VISIBLE_DEVICES=4
    exec python "$DIAG_CODE/env_info/a800_20260912/verify_online_fla_bridge.py"
    ;;
  dry-run)
    exec python "$DIAG_CODE/env_info/a800_20260912/run_fla_online_smoke.py" --dry-run
    ;;
  run)
    exec python "$DIAG_CODE/env_info/a800_20260912/run_fla_online_smoke.py"
    ;;
  *) exit 2 ;;
esac
