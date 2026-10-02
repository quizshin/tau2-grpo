"""Coverage-preserving capability slices and evidence-linked diagnostic badcases."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from tau3_grpo.evaluation.artifacts import read_evaluation
from tau3_grpo.evaluation.diagnostics import trajectory_diagnostics
from tau3_grpo.evaluation.rescore import rescore_run
from tau3_grpo.evaluation.rubric import identity, prepare, read_reviews
from tau3_grpo.evaluation.rubric_contract import DIMENSIONS, STATUSES, VERSION, write_json


def _slice(rows):
    dimensions = {}
    for dim in DIMENSIONS:
        counts = Counter(row['dimensions'][dim]['status'] if row['dimensions'] else 'unjudged' for row in rows)
        known = counts['satisfied'] + counts['violated']
        dimensions[dim] = {
            **{k: counts[k] for k in (*STATUSES, 'unjudged')},
            'known_verdict_satisfaction_rate': counts['satisfied'] / known if known else None,
        }
    return {'planned_trials': len(rows), 'dimensions': dimensions}


def build_report(run_dir, bundle, review_dir, *, retrospective=False):
    package = prepare(run_dir, bundle, retrospective=retrospective)
    manifest, reviews = read_reviews(review_dir, package)
    artifact = read_evaluation(run_dir)
    results = {identity(r): r for r in artifact.trajectories}
    requests = {identity(r): r for r in package['requests']}
    unavailable = {identity(r): r['reason'] for r in package['unavailable']}
    cases = []
    for job in package['planned']:
        key = identity(job)
        request = requests.get(key)
        response = reviews.get(request['request_sha256']) if request else None
        verdict = response['review']['dimensions'] if response and response['status'] == 'reviewed' else None
        result = results.get(key)
        success = abs(result['reward'] - 1) <= 1e-6 if result else None
        flags = []
        if success is False:
            flags.append('outcome_failure')
        if success is None:
            flags.append('unresolved_outcome')
        if verdict is None:
            flags.append('unjudged')
        elif any(v['status'] == 'violated' for v in verdict.values()):
            flags.append('rubric_violation_candidate')
        if verdict and any(v['status'] == 'unknown' for v in verdict.values()):
            flags.append('uncertain_judgment')
        task = bundle['tasks'][job['task_id']]
        cases.append({
            **job, 'outcome_success': success, 'dimensions': verdict, 'flags': flags,
            'request_sha256': request['request_sha256'] if request else None,
            'capabilities': task['capabilities'], 'bucket': task['bucket'],
            'unavailable_reason': unavailable.get(key),
            'judge_status': response['status'] if response else 'not_judged',
            'events': request['events'] if request else [],
        })
    complete = all(c['dimensions'] is not None for c in cases)
    tasks = {tid: _slice([c for c in cases if c['task_id'] == tid]) for tid in bundle['tasks']}
    macro = {}
    for dim in DIMENSIONS:
        rates = [t['dimensions'][dim]['known_verdict_satisfaction_rate'] for t in tasks.values()]
        macro[dim] = {
            'tasks_with_known_verdict': sum(r is not None for r in rates),
            'planned_tasks': len(tasks),
            # Partial known-only values are diagnostic; never headline as full coverage.
            'rate': sum(rates) / len(rates) if complete and all(r is not None for r in rates) else None,
        }
    return {
        'version': VERSION, 'bundle_sha256': bundle['bundle_sha256'],
        'package_sha256': package['package_sha256'], 'files': artifact.files,
        'pre_run_bound': package['pre_run_bound'], 'judge': manifest['judge'],
        'prompt_sha256': manifest['prompt_sha256'], 'judge_calibrated': False,
        'response_models': sorted({r['usage']['response_model'] for r in reviews.values()
                                   if r.get('usage') and r['usage'].get('response_model')}),
        'rubric_coverage_complete': complete, 'outcome': rescore_run(run_dir),
        'overall': _slice(cases), 'per_task': tasks, 'task_macro': macro,
        'capability_slices': {cap: _slice([c for c in cases if cap in c['capabilities']])
                              for cap in sorted({cap for c in cases for cap in c['capabilities']})},
        'curriculum_slices': {b: _slice([c for c in cases if c['bucket'] == b])
                             for b in sorted({c['bucket'] for c in cases})},
        'efficiency': trajectory_diagnostics(artifact.trajectories),
        'badcases': [c for c in cases if c['flags']],
        'limitations': [
            'Uncalibrated judge diagnostics, not official reward or validated capability accuracy.',
            'Unknown, not_applicable and unjudged are distinct; known-only rates are conditional.',
            'Task tags are reviewed in advance; overlapping capability slices are not additive.',
            'Evidence references validate identity, not semantic correctness or root cause.',
            'No summed rubric score; historical pilot scores cannot be compared to these verdicts.',
        ],
    }


def save_report(directory, report):
    import json

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'report.json', report)
    with (directory / 'badcases.jsonl').open('x') as handle:
        for row in report['badcases']:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    lines = ['# Five-dimension diagnostic evaluation', '',
             f"Rubric coverage complete: {report['rubric_coverage_complete']}",
             f"Outcome metrics valid: {report['outcome']['metrics_valid']}",
             f"Pre-run frozen rubric: {report['pre_run_bound']}", '',
             '| Dimension | Satisfied | Violated | Unknown | N/A | Unjudged |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for dim, counts in report['overall']['dimensions'].items():
        lines.append('| ' + dim + ' | ' + ' | '.join(str(counts[k]) for k in (*STATUSES, 'unjudged')) + ' |')
    lines += ['', 'Badcases and visible event evidence: `badcases.jsonl`.', '', *report['limitations']]
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')


def compare_reports(baseline, treatment, bundle, baseline_reviews, treatment_reviews):
    from tau3_grpo.evaluation.compare import compare_evaluations

    outcome = compare_evaluations(baseline, treatment)
    left = build_report(baseline, bundle, baseline_reviews)
    right = build_report(treatment, bundle, treatment_reviews)
    reasons = []
    if not outcome['comparable']:
        reasons.append('outcome_protocol_or_plan_incomplete')
    missing = set(outcome['protocol']['missing_extended_or_identity_evidence'])
    missing.discard('endpoints.policy.repetition_penalty')  # optional on legacy text harness
    if missing:
        reasons.append('missing_protocol_or_model_identity')
    for field in ('bundle_sha256', 'prompt_sha256', 'judge', 'response_models'):
        if left[field] != right[field]:
            reasons.append(f'different_{field}')
    if not all(r['rubric_coverage_complete'] and r['pre_run_bound'] for r in (left, right)):
        reasons.append('incomplete_or_retrospective_rubric')
    # Never compare conditional rates with a changed unknown/N/A denominator.
    for task_id in left['per_task']:
        for dim in DIMENSIONS:
            for status in ('unknown', 'not_applicable', 'unjudged'):
                if any(r['per_task'][task_id]['dimensions'][dim][status] for r in (left, right)):
                    reasons.append('non_decisive_verdicts')
    delta = None
    if not reasons:
        delta = {d: right['task_macro'][d]['rate'] - left['task_macro'][d]['rate'] for d in DIMENSIONS}
    return {'version': VERSION, 'comparable': not reasons, 'reasons': sorted(set(reasons)),
            'outcome': outcome, 'diagnostic_task_macro_delta': delta,
            'judge_calibrated': False,
            'note': 'Descriptive diagnostic difference; no claim of causal or statistically stable improvement.'}
