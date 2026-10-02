"""Frozen five-dimension trajectory evaluation: draft/freeze/prepare/judge/report.

Everything is offline except the explicitly selected judge command. The model
never receives rewards or reference trajectories. Training rewards are untouched.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from tau3_grpo.evaluation.artifacts import read_evaluation
from tau3_grpo.evaluation.rubric_contract import (
    DEFINITIONS,
    PROMPT,
    VERSION,
    bind_bundle,
    freeze_bundle,
    load_bundle,
    read_json,
    validate_review,
    visible_events,
    write_json,
)
from tau3_grpo.evaluation.scoring import summarize_trials
from tau3_grpo.utils.hashing import sha256_json


def identity(row):
    return tuple(row[k] for k in ('task_id', 'trial', 'seed'))


def schema_provenance():
    from tau3_grpo.envs.adapter import airline_tool_schemas

    return {'rubric_tool_schemas_sha256': sha256_json(airline_tool_schemas())}


def draft_bundle(jobs):
    """Generate review scaffolding, never claim these task criteria were reviewed."""
    from tau3_grpo.envs.adapter import airline_tool_schemas
    from tau3_grpo.evaluation.provenance import evaluation_provenance
    from tau3_grpo.prompts import build_system_prompt, prompt_provenance

    schemas = airline_tool_schemas()
    provenance = evaluation_provenance(jobs)
    return {
        'version': VERSION, 'policy': build_system_prompt(),
        'tool_schemas': schemas, 'tool_schemas_sha256': sha256_json(schemas),
        'agent_system_prompt_sha256': prompt_provenance()['agent_system_prompt_sha256'],
        'tasks': {j['task_id']: {
            'task_definition_sha256': provenance['task_definition_sha256'][j['task_id']],
            'task_db_sha256': provenance['task_db_sha256'][j['task_id']],
            'criteria': dict(DEFINITIONS), 'reviewed_by': '', 'capabilities': [],
            'bucket': 'unclassified',
        } for j in jobs},
    }


def prepare(run_dir, bundle, *, retrospective=False):
    artifact = read_evaluation(run_dir)
    meta = artifact.metadata
    # Full denominator validation, even if all rubric calls will be offline.
    summarize_trials(planned=meta['planned'], results=artifact.trajectories,
                     errors=artifact.errors, trials=meta['spec']['trials'], ks=meta['spec']['ks'])
    if bundle != freeze_bundle(bundle):
        raise ValueError('Unsealed rubric bundle')
    recorded = meta['provenance'].get('rubric_bundle_sha256')
    bind_bundle(bundle, meta['provenance'], [r['task_id'] for r in meta['planned']],
                allow_missing_schema=retrospective and not recorded)
    if recorded and recorded != bundle['bundle_sha256']:
        raise ValueError('Run was bound to a different rubric')
    if not recorded and not retrospective:
        raise ValueError('Historical run needs --retrospective; no pre-run rubric attestation')
    results = {identity(row): row for row in artifact.trajectories}
    requests, unavailable = [], []
    for job in meta['planned']:
        row = results.get(identity(job))
        key = {k: job[k] for k in ('task_id', 'trial', 'seed')}
        if row is None:
            unavailable.append({**key, 'reason': 'missing_or_infrastructure_error'})
            continue
        try:
            events = visible_events(row.get('simulation', {}).get('messages'))
        except ValueError:
            unavailable.append({**key, 'reason': 'missing_visible_trajectory'})
            continue
        request = {
            **key, 'version': VERSION, 'bundle_sha256': bundle['bundle_sha256'],
            'system': PROMPT, 'policy': bundle['policy'], 'tool_schemas': bundle['tool_schemas'],
            'criteria': bundle['tasks'][job['task_id']]['criteria'], 'events': events,
        }
        requests.append({**request, 'request_sha256': sha256_json(request)})
    body = {
        'version': VERSION, 'files': artifact.files, 'planned': meta['planned'],
        'bundle_sha256': bundle['bundle_sha256'], 'prompt_sha256': sha256_json(PROMPT),
        'pre_run_bound': bool(recorded), 'requests': requests, 'unavailable': unavailable,
    }
    return {**body, 'package_sha256': sha256_json(body)}


def validate_package(package):
    body = {k: v for k, v in package.items() if k != 'package_sha256'}
    if package.get('version') != VERSION or package.get('package_sha256') != sha256_json(body):
        raise ValueError('Request package changed')
    if package.get('prompt_sha256') != sha256_json(PROMPT):
        raise ValueError('Request prompt version differs')
    keys = []
    for req in package['requests']:
        body = {k: v for k, v in req.items() if k != 'request_sha256'}
        if req.get('request_sha256') != sha256_json(body) or req['system'] != PROMPT:
            raise ValueError('Request changed')
        if req['bundle_sha256'] != package['bundle_sha256']:
            raise ValueError('Request belongs to another bundle')
        keys.append(identity(req))
    keys += [identity(row) for row in package['unavailable']]
    plan = [identity(row) for row in package['planned']]
    if len(set(keys)) != len(keys) or len(set(plan)) != len(plan) or set(keys) != set(plan):
        raise ValueError('Package must preserve every planned identity exactly once')


async def judge(package, output, *, client, max_calls):
    """One attempt per request, durable reservations, no implicit retries/resume.

An interrupted output remains inspectable; report accepts completed response files
and leaves pending/unattempted requests unjudged. A rerun to this output is refused.
"""
    validate_package(package)
    if type(max_calls) is not int or max_calls <= 0 or len(package['requests']) > max_calls:
        raise ValueError('Explicit positive max_calls must cover this request package')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        'version': VERSION, 'package_sha256': package['package_sha256'],
        'judge': client.provenance, 'prompt_sha256': package['prompt_sha256'],
        'max_calls': max_calls, 'judge_calibrated': False,
    }
    write_json(output / 'manifest.json', manifest)
    for req in package['requests']:
        sha = req['request_sha256']
        write_json(output / f'{sha}.pending.json', {'request_sha256': sha, 'status': 'reserved'})
        transport = {'system': PROMPT, 'visible_messages': req['events'],
                     'prefix_sha256': sha256_json(req['events'])}
        packet = {'request_sha256': sha, 'status': 'invalid_or_failed', 'review': None}
        metadata_start = len(client.metadata)
        try:
            answer = await client.extract_json(transport, user_payload={
                k: req[k] for k in ('policy', 'tool_schemas', 'criteria', 'events')
            })
            packet.update(status='reviewed', review=validate_review(answer, req))
        except Exception as error:
            # Never publish provider bodies, credentials or exception messages.
            packet['error_type'] = type(error).__name__
        packet['usage'] = (client.metadata[-1] if len(client.metadata) > metadata_start else None)
        write_json(output / f'{sha}.response.json', packet)
    return manifest


def read_reviews(directory, package):
    directory = Path(directory)
    manifest = read_json(directory / 'manifest.json')
    if manifest.get('version') != VERSION or manifest.get('package_sha256') != package['package_sha256']:
        raise ValueError('Reviews belong to different requests')
    if manifest.get('prompt_sha256') != package['prompt_sha256'] or not manifest.get('judge'):
        raise ValueError('Judge identity or prompt evidence missing')
    requests = {r['request_sha256']: r for r in package['requests']}
    reviews = {}
    for path in sorted(directory.glob('*.response.json')):
        row = read_json(path)
        sha = row['request_sha256']
        if sha not in requests or sha in reviews:
            raise ValueError('Unknown or duplicate rubric response')
        if row['status'] == 'reviewed':
            validate_review(row['review'], requests[sha])
        elif row['status'] != 'invalid_or_failed' or row.get('review') is not None:
            raise ValueError('Malformed failed rubric response')
        reviews[sha] = row
    return manifest, reviews


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    draft = subs.add_parser('draft', help='Create unreviewed task criteria; no model calls')
    draft.add_argument('--manifest', type=Path, required=True)
    draft.add_argument('--output', type=Path, required=True)
    freeze = subs.add_parser('freeze', help='Seal reviewer-approved task criteria')
    freeze.add_argument('--draft', type=Path, required=True)
    freeze.add_argument('--output', type=Path, required=True)
    for command in ('prepare', 'report'):
        sub = subs.add_parser(command)
        sub.add_argument('--run-dir', type=Path, required=True)
        sub.add_argument('--bundle', type=Path, required=True)
        sub.add_argument('--retrospective', action='store_true')
        sub.add_argument('--output', type=Path, required=True)
        if command == 'report':
            sub.add_argument('--reviews', type=Path, required=True)
    online = subs.add_parser('judge', help='Explicit online judge; incurs API usage')
    online.add_argument('--requests', type=Path, required=True)
    online.add_argument('--output', type=Path, required=True)
    online.add_argument('--max-calls', type=int, required=True)
    online.add_argument('--max-tokens', type=int, default=4096)
    online.add_argument('--execute', action='store_true', required=True)
    compare = subs.add_parser('compare')
    for flag in ('baseline', 'treatment', 'baseline-reviews', 'treatment-reviews', 'bundle', 'output'):
        compare.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'draft':
        from tau3_grpo.data.manifest import read_manifest
        from tau3_grpo.evaluation.runtime import _selection_jobs

        jobs = list(_selection_jobs(read_manifest(args.manifest), 1, 42))
        write_json(args.output, draft_bundle(jobs))
    elif args.command == 'freeze':
        write_json(args.output, freeze_bundle(read_json(args.draft)))
    elif args.command == 'prepare':
        write_json(args.output, prepare(args.run_dir, load_bundle(args.bundle), retrospective=args.retrospective))
    elif args.command == 'judge':
        from tau3_grpo.models.semantic_api import OpenAICompatibleSemanticModel

        package = read_json(args.requests)
        validate_package(package)
        client = OpenAICompatibleSemanticModel.from_env(
            {'max_tokens': args.max_tokens, 'temperature': 0, 'response_format_json': True},
            env_prefix='TAU3_RUBRIC',
        )
        asyncio.run(judge(package, args.output, client=client, max_calls=args.max_calls))
        _, responses = read_reviews(args.output, package)
        return 0 if all(r['status'] == 'reviewed' for r in responses.values()) else 1
    elif args.command == 'report':
        from tau3_grpo.evaluation.rubric_report import build_report, save_report

        report = build_report(args.run_dir, load_bundle(args.bundle), args.reviews,
                              retrospective=args.retrospective)
        save_report(args.output, report)
        return 0 if report['rubric_coverage_complete'] and report['outcome']['metrics_valid'] else 1
    else:
        from tau3_grpo.evaluation.rubric_report import compare_reports

        comparison = compare_reports(args.baseline, args.treatment, load_bundle(args.bundle),
                                     args.baseline_reviews, args.treatment_reviews)
        write_json(args.output, comparison)
        return 0 if comparison['comparable'] else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
