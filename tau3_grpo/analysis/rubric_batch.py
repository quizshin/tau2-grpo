"""Retired CLI; import forwarding for historical replay only."""
from tau3_grpo.data.messages import visible_events as visible_events
from tau3_grpo.tracking.judge_budget import Budget as Budget
from tau3_grpo.tracking.judge_budget import call_json as call_json
from tau3_grpo.tracking.judge_budget import dump as dump
from tau3_grpo.tracking.judge_budget import flash_usage_estimate as flash_usage_estimate


def __getattr__(name):
    from tau3_grpo.analysis.legacy import rubric_batch

    return getattr(rubric_batch, name)


if __name__ == "__main__":
    raise SystemExit("Retired rubric pilot. Use python -m tau3_grpo.evaluation.rubric --help. "
                     "Historical replay only: tau3_grpo.analysis.legacy.rubric_batch --legacy-review")
