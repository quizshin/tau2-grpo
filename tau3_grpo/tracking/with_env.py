"""Load code/.env without shell evaluation, then exec a training command."""

import os
import sys

from tau3_grpo.tracking.swanlab import load_tracking_env

if __name__ == "__main__":
    load_tracking_env()
    os.environ["TAU3_TRACKING_ENV_LOADED"] = "1"
    os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
