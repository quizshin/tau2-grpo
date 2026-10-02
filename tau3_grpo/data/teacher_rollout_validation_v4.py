"""Historical validation entry; implementation lives in teacher_rollout."""

from functools import partial
from pathlib import Path

from tau3_grpo.data import teacher_rollout as _core

validate_task = partial(_core.validate_task, split="validation")
generate_candidate = partial(_core.generate_candidate, split="validation")
IncrementBudget = partial(_core.IncrementBudget, limit_cny=150.0)
run = partial(_core.run, default_split="validation", default_limit_cny=150.0,
              entrypoint=Path(__file__))


def __getattr__(name):
    return getattr(_core, name)


def main(argv=None):
    _core.main(argv, default_split="validation", default_limit_cny=150.0,
               entrypoint=Path(__file__))


if __name__ == "__main__":
    main()
