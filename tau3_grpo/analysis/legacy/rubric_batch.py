"""Versioned review batches sharing the original cumulative DeepSeek budget ledger."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
from collections import Counter
from pathlib import Path

from tau3_grpo.analysis.legacy.rubric_pilot import (
    Budget,
    call_json,
    dump,
    prompts_v2,
    validate_audit,
    validate_rubric,
    visible_events,
)
from tau3_grpo.models.semantic_api import SemanticAPIError
from tau3_grpo.utils.hashing import sha256_file, sha256_json


async def run(args):
    import yaml

    args.output.mkdir(parents=True, exist_ok=True)
    # Same lock as v1: a second runner cannot spend against a stale ledger.
    with (args.budget_directory / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = Budget(args.budget_directory / 'budget.json', 100, max_calls=2000)
        rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
        ids = [r['metadata']['source_dialog_id'] for r in rows]
        if len(set(ids)) != len(ids) or not rows or len(rows) > 250:
            raise ValueError('Expected 1-250 uniquely identified review records')
        schemas = [x['tool_schema'] for x in yaml.safe_load(args.tools.read_text())['tools']]
        author, audit_prompt = prompts_v2()
        manifest = dict(version='airline_rubric_v2', batch=args.batch, input_sha256=sha256_file(args.input),
                        schema_sha256=sha256_json(schemas), author_sha256=sha256_json(author),
                        audit_sha256=sha256_json(audit_prompt), ids=ids,
                        ledger=str((args.budget_directory / 'budget.json').resolve()), review_only=True)
        dest = args.output / 'manifest.json'
        if dest.exists() and json.loads(dest.read_text()) != manifest:
            raise ValueError('Immutable batch identity changed')
        dump(dest, manifest)
        for folder in ('calls', 'records'):
            (args.output / folder).mkdir(exist_ok=True)
        sem = asyncio.Semaphore(4)
        results = []

        async def one(row):
            async with sem:
                sid = row['metadata']['source_dialog_id']
                path = args.output / 'records' / f'{sid}.json'
                if path.exists():
                    results.append(json.loads(path.read_text()))
                    return
                events = visible_events(row['messages'])
                context = [e for e in events if e['role'] != 'assistant']
                result = dict(source_dialog_id=sid, visible_hash=sha256_json(events),
                              status='not_judged', review_only=True)
                try:
                    prefix = f'v2_{args.batch}_{sid}'
                    rubric = await call_json(args.output, budget, prefix + '_rubric', author, context,
                                             {'events': context, 'tool_schemas': schemas})
                    validate_rubric(rubric, context)
                    result['rubric'] = rubric
                    audit = await call_json(args.output, budget, prefix + '_audit', audit_prompt, events,
                                            {'events': events, 'tool_schemas': schemas, 'rubric': rubric})
                    validate_audit(audit, rubric, events)
                    if audit['recommendation'] == 'keep_candidate' and (
                            audit['issues'] or any(r['status'] in ('unknown', 'violated') for r in audit['requirements'])):
                        raise ValueError('Keep recommendation conflicts with unresolved/violated requirements')
                    result.update(status='review_ready', audit=audit)
                except (SemanticAPIError, ValueError, TypeError, KeyError, AttributeError) as exc:
                    result.update(error_type=type(exc).__name__, error=str(exc)[:500])
                dump(path, result)
                results.append(result)
                print(json.dumps({'done': len(results), 'total': len(rows), 'id': sid,
                                  'status': result['status'], 'cumulative_cny': budget.accounted}), flush=True)

        await asyncio.gather(*(one(r) for r in rows))
        summary = dict(planned=len(rows), completed=len(results),
                       statuses=dict(Counter(r['status'] for r in results)),
                       recommendations=dict(Counter(r['audit']['recommendation'] for r in results if 'audit' in r)),
                       cumulative_cny=budget.accounted, cumulative_calls=len(budget.state['calls']),
                       automatic_training_acceptance=False, official_replay=False)
        dump(args.output / 'summary.json', summary)
        print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch', required=True)
    p.add_argument('--budget-directory', type=Path,
                   default=Path('results/analysis/deepseek_rubric_pilot_20260925'))
    p.add_argument('--tools', type=Path, default=Path('configs/envs/tool_config.yaml'))
    args = p.parse_args()
    if not args.batch.replace('_', '').isalnum():
        p.error('Batch must be an alphanumeric namespace')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
