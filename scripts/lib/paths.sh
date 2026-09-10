#!/usr/bin/env bash
# Runtime storage may live outside the source checkout.
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="${CODE_ROOT}"
export TAU3_DATA_ROOT="${TAU3_DATA_ROOT:-${CODE_ROOT}/data}"
export TAU3_MODEL_ROOT="${TAU3_MODEL_ROOT:-${CODE_ROOT}/models}"
export TAU3_RUN_ROOT="${TAU3_RUN_ROOT:-${CODE_ROOT}/results}"
export TAU3_CACHE_ROOT="${TAU3_CACHE_ROOT:-${CODE_ROOT}/.cache}"
