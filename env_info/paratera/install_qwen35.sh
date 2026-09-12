#!/usr/bin/env bash
set -euo pipefail
source /root/shared-nvme/tau3/code/env_info/paratera/activate.sh
# Run under run_guarded.py. Retain completed downloads across network retries.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-3600}"
export UV_HTTP_RETRIES=8
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-12}"
export TAU3_PYPI_INDEX="${TAU3_PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
python -m pip install --no-cache-dir --index-url "${TAU3_PYPI_INDEX}" uv==0.12.12
WHEELS=()
if [[ -n "${TAU3_WHEELHOUSE:-}" ]]; then
  for wheel in "${TAU3_WHEELHOUSE}"/torch-2.11.0-*.whl "${TAU3_WHEELHOUSE}"/vllm-0.20.0-*.whl; do
    test -f "${wheel}"
    WHEELS+=("${wheel}")
  done
fi
"${TAU3_UV}" pip install --python "${TAU3_BASE_VENV}/bin/python" \
  --index-url "${TAU3_PYPI_INDEX}" \
  --constraint "${TAU3_ROOT}/code/env_info/qwen35-constraints.txt" \
  --constraint "${TAU3_ROOT}/code/env_info/paratera/constraints.txt" \
  -e "${TAU3_ROOT}/code/tau2-bench" \
  -e "${TAU3_ROOT}/code/verl[qwen35]" \
  -e "${TAU3_ROOT}/code[dev,data,stats,qwen35,tracking]" "${WHEELS[@]}"
python -m pip check
python -m pip freeze > "${TAU3_ROOT}/bootstrap/qwen35-installed.txt"
