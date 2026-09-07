"""Command-line entrypoints.

Each module is runnable as `python -m tau3_grpo.cli.<name>` and is also exposed as
a console script in `pyproject.toml`. Modules are not imported here because
`build_parquet` pulls in veRL.
"""

__all__ = [
    "build_parquet",
    "attest_service",
    "evaluate",
    "freeze_winner",
    "prepare_data",
    "prepare_experiment",
    "prepare_sft",
    "report",
    "sft_train",
    "verify_patches",
]
