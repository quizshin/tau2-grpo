"""Start veRL after registering Tau-GiGPO in the same Python process."""

from __future__ import annotations

import os
import runpy


def main() -> int:
    # The estimator registry is process-local. A preliminary `python -c`
    # process cannot register an estimator for the subsequent trainer process.
    from tau3_grpo.integrations.verl.gigpo import register

    register()
    from tau3_grpo.integrations.verl.mt_gtpo import register as register_mt_gtpo

    register_mt_gtpo()
    # Rollout workers resolve this hook lazily on their first assistant segment.
    os.environ.setdefault(
        "TAU3_GRPO_ANCHOR_HOOK",
        "tau3_grpo.integrations.anchor_hook:current_anchor",
    )
    runpy.run_module("verl.trainer.main_ppo", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
