"""veRL integration: interaction, tools, session registry, patch contract.

`interaction` and `tools` import veRL at module import time, so they are not
re-exported here; importing `tau3_grpo.integrations` must stay safe on a machine
without veRL installed. Import the submodules directly instead.
"""

from tau3_grpo.envs.registry import SESSIONS, SessionEntry, SessionRegistry, session_for
from tau3_grpo.integrations.patch_contract import (
    PATCH_MARKER,
    REQUIREMENTS,
    PatchRequirement,
    assert_patched,
    check_patch,
    vanilla_grpo_untouched,
    verify_all,
)

__all__ = [
    "PATCH_MARKER",
    "REQUIREMENTS",
    "SESSIONS",
    "PatchRequirement",
    "SessionEntry",
    "SessionRegistry",
    "assert_patched",
    "check_patch",
    "session_for",
    "vanilla_grpo_untouched",
    "verify_all",
]

# `anchor_hook`, `interaction` and `tools` import veRL, so they stay out of the
# eager import list above. Import them directly where veRL is installed.
