#!/usr/bin/env bash
# Formal four-arm controller using the previously validated FLA-only overlay.
set -euo pipefail
MATCHED_CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /root/autodl-fs/tau3-core/activate.sh
export TAU3_ENV_FILE="$TAU3_ROOT/code/.env"
export PYTHONPATH="$MATCHED_CODE:$MATCHED_CODE/verl:$MATCHED_CODE/tau2-bench/src:/root/autodl-fs/tau3-core/environment/overlays/fla"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$MATCHED_CODE"
case "${1:-run}" in
  e1-dry-run) exec python env_info/a800_20260912/run_e1_after_e0.py --dry-run ;;
  e1) exec python env_info/a800_20260912/run_e1_after_e0.py ;;
  dry-run) exec python env_info/a800_20260912/run_matched50.py --dry-run ;;
  run) exec python env_info/a800_20260912/run_matched50.py ;;
  *) exit 2 ;;
esac
