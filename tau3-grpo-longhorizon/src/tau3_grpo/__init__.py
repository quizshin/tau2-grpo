"""τ³-GRPO long-horizon: AReaL τ²-style training, τ³ official final evaluation.

Data boundary (immutable):

- Training and internal selection come from `inclusionAI/AReaL-tau2-data`
  revision 86971dc03da6e7c1a7933295e05b84aab8215386: 1,982 records, 1,148 Airline,
  frozen into 200 train / 60 selection / 888 reserve.
- AReaL is τ²-style synthetic data. Converting its schema does not make it τ³
  official, and it is never labelled as such.
- The τ³ official final is the tau2-bench v1.0.1 Airline `base` split (50 tasks).
  It runs only after the winner lock is frozen, and never for training or model
  selection.

Submodules that require veRL (`integration.interaction`, `integration.tools`,
`algo.verl_estimator`, `data.parquet_builder`) are not imported here so the pure
CPU test suite runs without veRL installed.
"""

from tau3_grpo.paths import (
    CODE_ROOT,
    ENV_INFO_ROOT,
    PROJECT_ROOT,
    TAU2_BENCH_ROOT,
    VERL_ROOT,
    verify_four_root_layout,
)

__version__ = "0.1.0"

AREAL_REVISION = "86971dc03da6e7c1a7933295e05b84aab8215386"
TAU2_BENCH_COMMIT = "fc0055d"
TAU2_BENCH_VERSION = "1.0.1"
VERL_COMMIT = "bec9ef7"
VERL_VERSION = "0.7.1"

__all__ = [
    "AREAL_REVISION",
    "CODE_ROOT",
    "ENV_INFO_ROOT",
    "PROJECT_ROOT",
    "TAU2_BENCH_COMMIT",
    "TAU2_BENCH_VERSION",
    "VERL_COMMIT",
    "VERL_VERSION",
    "TAU2_BENCH_ROOT",
    "VERL_ROOT",
    "__version__",
    "verify_four_root_layout",
]
