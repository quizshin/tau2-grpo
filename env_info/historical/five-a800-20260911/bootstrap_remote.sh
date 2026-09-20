#!/usr/bin/env bash
# Run only on the platform-verified newly rented five-A800 instance.
set -euo pipefail
export PATH="/root/miniconda3/bin:$PATH"
export PIP_NO_CACHE_DIR=1
DEPLOY_ROOT="${TAU3_DEPLOY_ROOT:-/root/autodl-tmp/tau3-5xa800-20260911}"
CODE_ROOT="${DEPLOY_ROOT}/code"
VENV_ROOT="${DEPLOY_ROOT}/venvs/qwen35"
mkdir -p "${DEPLOY_ROOT}/cache" "${DEPLOY_ROOT}/tmp" "${DEPLOY_ROOT}/validation"
export PIP_CACHE_DIR="${DEPLOY_ROOT}/cache/pip"
export TMPDIR="${DEPLOY_ROOT}/tmp"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="${DEPLOY_ROOT}/cache/huggingface"
export TAU3_ROOT="${DEPLOY_ROOT}"
export TAU3_BASE_VENV="${VENV_ROOT}"
export TAU3_SIM_VENV="${DEPLOY_ROOT}/venvs/qwen38-sim"
test -f "${CODE_ROOT}/pyproject.toml"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "${DEPLOY_ROOT}/validation/gpus.csv"
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
[[ "${GPU_COUNT}" -eq 5 ]] || { echo 'Expected exactly five visible GPUs'; exit 2; }
PYTHON_BIN="$(command -v python3.12 || command -v python3)"
"${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
cd "${CODE_ROOT}"
TAU3_VENV_DIR="${VENV_ROOT}" PYTHON_BIN="${PYTHON_BIN}" bash setup.sh a800-qwen35
"${VENV_ROOT}/bin/python" -m pip install 'uv==0.12.12'
export TAU3_UV="${VENV_ROOT}/bin/uv"
bash env_info/setup_qwen38_simulator.sh
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/tau2-bench/src:${CODE_ROOT}/verl"
"${VENV_ROOT}/bin/python" -m pip check
"${VENV_ROOT}/bin/python" -m tau3_grpo.integrations.verify_patches
"${VENV_ROOT}/bin/python" - <<'PY' | tee "${DEPLOY_ROOT}/validation/cuda-check.json"
import json
from importlib.metadata import version
import torch
import vllm._C
assert version('torch').split('+')[0] == '2.11.0'
assert version('vllm') == '0.20.0'
assert version('transformers') == '5.5.1'
assert torch.version.cuda == '13.0', torch.version.cuda
assert torch.cuda.device_count() == 5
rows=[]
for i in range(5):
    assert 'A800' in torch.cuda.get_device_name(i)
    with torch.cuda.device(i):
        x=torch.randn(32,32,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        y=(x@x).float().square().mean()
        y.backward()
        torch.cuda.synchronize()
        assert torch.isfinite(x.grad).all()
        rows.append({'index':i,'name':torch.cuda.get_device_name(i),'bf16_backward':True})
print(json.dumps({'versions':{k:version(k) for k in ['torch','vllm','transformers','peft','accelerate','swanlab']},'cuda':torch.version.cuda,'gpus':rows},indent=2))
PY
cat > "${DEPLOY_ROOT}/validation/nccl_smoke.py" <<'PYCODE'
import os
import datetime
import torch
import torch.distributed as dist
rank=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=90))
x=torch.tensor([rank+1.0],device='cuda')
dist.all_reduce(x)
assert x.item() == 15.0, x.item()
dist.barrier()
dist.destroy_process_group()
if rank == 0: print('NCCL five-GPU all_reduce passed')
PYCODE
timeout 180 "${VENV_ROOT}/bin/torchrun" --standalone --nproc-per-node=5 "${DEPLOY_ROOT}/validation/nccl_smoke.py" | tee "${DEPLOY_ROOT}/validation/nccl.log"
"${VENV_ROOT}/bin/python" -m pytest -q tests/test_launcher.py tests/test_entropy_memory.py
"${VENV_ROOT}/bin/python" -m pip freeze > "${DEPLOY_ROOT}/validation/installed.txt"
echo 'Dependency and per-GPU validation complete. Models, data, NCCL and launch dry-runs must be verified separately before declaring ready.'
