"""Separate completed attempts with invalid metrics from a crashed evaluation."""
import json
from pathlib import Path


def classify_completion(output_dir, returncode):
    out = Path(output_dir)
    if returncode not in (0, 1):
        raise RuntimeError(f'evaluation process failed: exit={returncode}')
    try:
        plan = json.loads((out / 'run.json').read_text())['planned']
        summary = json.loads((out / 'summary.json').read_text())
        def rows(name):
            return [json.loads(line) for line in (out / name).read_text().splitlines() if line.strip()]
        results, errors = rows('trajectories.jsonl'), rows('errors.jsonl')
        def key(row):
            return row['task_id'], row['trial'], row['seed']
        expected = [key(row) for row in plan]
        actual = [key(row) for row in results + errors]
        if not expected or len(set(expected)) != len(expected):
            raise ValueError('invalid plan')
        if len(actual) != len(expected) or set(actual) != set(expected):
            raise ValueError('missing, duplicate, or unexpected attempts')
        for field, value in [('planned_trajectories', len(plan)), ('completed_trajectories', len(results)), ('failed_trajectories', len(errors)), ('missing_trajectories', 0)]:
            if summary[field] != value:
                raise ValueError(f'inconsistent summary: {field}')
        if summary['metrics_valid'] != (not errors) or returncode != (1 if errors else 0):
            raise ValueError('exit code / metric validity mismatch')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f'evaluation did not finish reliably: {exc}') from exc
    return {'status': 'completed_with_errors' if errors else 'complete',
            'evaluation_exit': returncode, 'metrics_valid': summary['metrics_valid'],
            'scored_attempts': len(results), 'error_attempts': len(errors),
            'error_types': sorted({row.get('error_type', 'unknown') for row in errors})}
