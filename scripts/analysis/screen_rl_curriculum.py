"""Compatibility command; implementation is versioned inside tau3_grpo.data."""

from tau3_grpo.data import rl_curriculum_screen as _core

main = _core.main


def __getattr__(name):
    return getattr(_core, name)


if __name__ == "__main__":
    main()
