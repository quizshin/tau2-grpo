"""Compatibility import for historical callers; implementation lives in integrations.verl.gigpo.

Module identity is shared so registry and last-batch statistics cannot diverge.
Remove only after all supported historical entrypoints have migrated.
"""

import sys

from tau3_grpo.integrations.verl import gigpo as _implementation

sys.modules[__name__] = _implementation
