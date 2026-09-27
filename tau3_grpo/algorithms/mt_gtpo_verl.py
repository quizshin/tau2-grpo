"""Compatibility import for historical callers; implementation lives in integrations.verl.mt_gtpo.

Module identity is shared so registry and last-batch statistics cannot diverge.
Remove only after all supported historical entrypoints have migrated.
"""

import sys

from tau3_grpo.integrations.verl import mt_gtpo as _implementation

sys.modules[__name__] = _implementation
