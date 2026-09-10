#!/usr/bin/env bash
# Internal selection (60 tasks) or the τ³ official final (50 tasks).
#
# Milestone D13. The τ³ final path refuses to start without a frozen winner lock
# naming and hashing this exact merged checkpoint; the real run also requires
# the live vLLM service attestation written by serve_policy_eval.sh.
#
# Usage:
#   scripts/eval/run.sh selection results/e3_seed42/global_step_40/merged_hf 42
#   scripts/eval/run.sh tau3-final results/e3_seed42/global_step_40/merged_hf

set -euo pipefail

TARGET="${1:?usage: run_eval.sh <selection|tau3-final> <checkpoint> [seed]}"
CHECKPOINT="${2:?checkpoint path required}"
SEED="${3:-42}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODE_ROOT="${PROJECT_ROOT}"
source "${CODE_ROOT}/scripts/lib/paths.sh"

export PYTHONPATH="${PROJECT_ROOT}:${CODE_ROOT}/tau2-bench/src:${CODE_ROOT}/verl:${PYTHONPATH:-}"

case "${TARGET}" in
  selection|tau3-final) ;;
  *) echo "error: target must be 'selection' or 'tau3-final'" >&2; exit 2 ;;
esac

# Resolves the task set and enforces the winner lock for tau3-final.
python -m tau3_grpo.evaluation.run \
  --target "${TARGET}" \
  --checkpoint "${CHECKPOINT}" \
  --seed "${SEED}" \
  --dry-run

if [[ "${TARGET}" == "tau3-final" ]]; then
  echo
  echo "winner lock verified. τ³ official final is a single frozen run:"
  echo "  - 50 Airline base tasks"
  echo "  - eval temperature 0.4"
  echo "  - no model selection may depend on this result"
fi

EVAL_TEMP=0.4
MAX_USER_TURNS=15
MAX_ASSISTANT_TURNS=15
EVAL_TRIALS="${EVAL_TRIALS:-4}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-16}"
POLICY_BASE_URL="${TAU3_POLICY_BASE_URL:-http://127.0.0.1:8000/v1}"
USER_BASE_URL="${TAU3_USER_BASE_URL:-http://127.0.0.1:8100/v1}"
POLICY_MODEL="${TAU3_POLICY_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
USER_MODEL="${TAU3_USER_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

echo
echo "launch rollouts with the served policy at ${CHECKPOINT}"
echo "eval temperature=${EVAL_TEMP} max_turns=${MAX_USER_TURNS}/${MAX_ASSISTANT_TURNS}"
echo "policy=${POLICY_BASE_URL} user=${USER_BASE_URL} trials=${EVAL_TRIALS}"

# Run real tau2 orchestrators and the official evaluator. For selection, each
# manifest record supplies its own FlightDB. For tau3-final, the winner lock
# above gates the official 50-task Airline base split.
python -m tau3_grpo.evaluation.run \
  --target "${TARGET}" \
  --checkpoint "${CHECKPOINT}" \
  --seed "${SEED}" \
  --policy-base-url "${POLICY_BASE_URL}" \
  --policy-model "${POLICY_MODEL}" \
  --user-base-url "${USER_BASE_URL}" \
  --user-model "${USER_MODEL}" \
  --trials "${EVAL_TRIALS}" \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --max-steps "$((MAX_USER_TURNS + MAX_ASSISTANT_TURNS))"
