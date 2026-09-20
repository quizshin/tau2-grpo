"""Historical MT-GTPO command; shared implementation is training.rl.runner."""

from tau3_grpo.training.rl import runner as _runner


def __getattr__(name):
    return getattr(_runner, name)


if __name__ == "__main__":
    _runner.main()
