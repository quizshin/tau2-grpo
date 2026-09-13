#!/usr/bin/env bash
set -euo pipefail
POST_CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /root/autodl-fs/tau3-core-20260912/activate.sh
export PYTHONPATH="$POST_CODE:$POST_CODE/verl:$POST_CODE/tau2-bench/src:/root/autodl-tmp/tau3-perf-20260912/overlay"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1 TAU3_GRPO_TEXT_ONLY=1
cd "$POST_CODE"
exec python env_info/a800_20260912/run_post_rl_eval.py \
  --root "$TAU3_RUN_ROOT/rl-c50-matched6h-a800-20260912" \
  --output "$TAU3_RUN_ROOT/post-rl-selection-20260914" --mode "${1:-dry-run}"
