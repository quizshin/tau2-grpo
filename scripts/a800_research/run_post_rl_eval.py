"""Compatibility entry for the packaged independent evaluation controller."""
import sys
from tau3_grpo.evaluation import controller as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
sys.modules[__name__] = _implementation
# Expose names to callers using importlib.exec_module on the original object.
globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith("__")})
