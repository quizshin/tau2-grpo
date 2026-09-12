"""Choose one shared update budget from E0's observed elapsed time.

This is an opt-in soft limit: finish the next multiple of the checkpoint/eval
interval, then use that exact update count for all comparison arms.
"""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time


@dataclass
class MatchedBudget:
    started_at: float
    budget_seconds: float
    interval: int = 10
    target_step: int | None = None
    observed_step: int | None = None
    state_path: Path | None = None

    def __post_init__(self):
        if not math.isfinite(self.started_at) or not math.isfinite(self.budget_seconds):
            raise ValueError('Budget times must be finite')
        if self.budget_seconds <= 0 or self.interval <= 0:
            raise ValueError('Budget and interval must be positive')

    @classmethod
    def from_environment(cls, *, save_freq, test_freq):
        seconds = os.environ.get('TAU3_E0_DISCOVERY_SECONDS')
        if not seconds:
            return None
        if os.environ.get('TAU3_GRPO_ARM') != 'e0':
            raise ValueError('Only E0 may discover the shared step budget')
        interval = int(os.environ.get('TAU3_BUDGET_INTERVAL', '10'))
        if save_freq != interval or test_freq != interval:
            raise ValueError('Budget boundaries must match checkpoint and evaluation intervals')
        return cls(started_at=float(os.environ['TAU3_BUDGET_STARTED_AT']),
                   budget_seconds=float(seconds), interval=interval,
                   state_path=Path(os.environ['TAU3_BUDGET_STATE_PATH']))

    def observe(self, step, *, now=None, ceiling=None):
        if step < 1:
            raise ValueError('Observe only completed positive training steps')
        elapsed = max(0.0, (time.time() if now is None else now) - self.started_at)
        if ceiling is not None and (ceiling < step or ceiling % self.interval):
            raise ValueError('Discovery ceiling must be a future budget boundary')
        if self.target_step is None and (elapsed >= self.budget_seconds or step == ceiling):
            self.target_step = ((step + self.interval - 1) // self.interval) * self.interval
            self.observed_step = step
            if ceiling is not None:
                if ceiling % self.interval:
                    raise ValueError('Discovery ceiling must end at a budget boundary')
                self.target_step = min(self.target_step, ceiling)
        state = {'mode': 'e0_time_then_round_up_shared_steps', 'completed_step': step,
                 'elapsed_seconds': elapsed, 'requested_seconds': self.budget_seconds,
                 'observed_step_at_threshold': self.observed_step, 'target_step': self.target_step,
                 'interval': self.interval, 'includes_startup_save_and_eval': True,
                 'soft_limit': True}
        if self.state_path is not None:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, indent=2))
            temporary.replace(self.state_path)
        return self.target_step is not None and step >= self.target_step
