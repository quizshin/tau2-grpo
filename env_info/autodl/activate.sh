#!/usr/bin/env bash
# Source after mounting the persistent volume at the same path.
export TAU3_FS_ROOT="${TAU3_FS_ROOT:-/root/autodl-fs/tau3_grpo_fix}"
export TAU3_ROOT="${TAU3_FS_ROOT}"
export TAU3_SCRATCH_ROOT="${TAU3_SCRATCH_ROOT:-/root/autodl-tmp/tau3}"
_tau3_venv="${TAU3_FS_ROOT}/runtime/venvs/qwen35"
export TAU3_CODE_ROOT="${TAU3_FS_ROOT}/code"
if [[ -f "${TAU3_SCRATCH_ROOT}/runtime/code/pyproject.toml" ]]; then
  export TAU3_CODE_ROOT="${TAU3_SCRATCH_ROOT}/runtime/code"
fi
if [[ -x "${TAU3_SCRATCH_ROOT}/runtime/qwen35/bin/python" && -f "${TAU3_SCRATCH_ROOT}/runtime/code/pyproject.toml" ]]; then
  _tau3_venv="${TAU3_SCRATCH_ROOT}/runtime/qwen35"
  export TAU3_CODE_ROOT="${TAU3_SCRATCH_ROOT}/runtime/code"
fi
source "${_tau3_venv}/bin/activate" || return 1
# Copied venv activation files may still embed the persistent prefix.
export VIRTUAL_ENV="${_tau3_venv}"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
unset _tau3_venv
export PATH="${TAU3_FS_ROOT}/runtime/bin:${PATH}"
export PYTHONPATH="${TAU3_CODE_ROOT}:${TAU3_CODE_ROOT}/tau2-bench/src:${TAU3_CODE_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"
export TAU3_DATA_ROOT="${TAU3_FS_ROOT}/data"
[[ ! -d "${TAU3_SCRATCH_ROOT}/data/raw" ]] || export TAU3_DATA_ROOT="${TAU3_SCRATCH_ROOT}/data"
export TAU3_MODEL_ROOT="${TAU3_FS_ROOT}/model_store"
[[ ! -f "${TAU3_SCRATCH_ROOT}/models/.tau3_models_verified.json" ]] || export TAU3_MODEL_ROOT="${TAU3_SCRATCH_ROOT}/models"
export TAU3_RUN_ROOT="${TAU3_SCRATCH_ROOT}/runs"
export TAU3_CACHE_ROOT="${TAU3_SCRATCH_ROOT}/cache"
export XDG_CACHE_HOME="${TAU3_CACHE_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX
export HF_HOME="${TAU3_CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PIP_CACHE_DIR="${TAU3_CACHE_ROOT}/pip"
export UV_CACHE_DIR="${TAU3_CACHE_ROOT}/uv"
export RUFF_CACHE_DIR="${TAU3_CACHE_ROOT}/ruff"
export VLLM_CACHE_ROOT="${TAU3_CACHE_ROOT}/vllm"
export TRITON_CACHE_DIR="${TAU3_CACHE_ROOT}/triton"
export TORCH_HOME="${TAU3_CACHE_ROOT}/torch"
export TORCHINDUCTOR_CACHE_DIR="${TAU3_CACHE_ROOT}/torchinductor"
export TAU3_SIM_CACHE_ROOT="${TAU3_CACHE_ROOT}/qwen38-sim"
export TMPDIR="${TAU3_SCRATCH_ROOT}/tmp"
export RAY_TMPDIR="${TMPDIR}/ray"
export WANDB_DIR="${TAU3_RUN_ROOT}/wandb"
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-${TAU3_FS_ROOT}/results/swanlog}"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" "${TAU3_CACHE_ROOT}" "${TAU3_RUN_ROOT}" "${SWANLAB_LOG_DIR}"
