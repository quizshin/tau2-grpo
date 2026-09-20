"""Audit v2 rule scope on the training pool and historical training calls only.

Historical text lacks reliable generation spans. We replay parsed calls into
fresh DBs and require call counts and initial/final DB hashes to agree. This
checks call-level reward eligibility, not turn advantages or user authorization.
No selection/final trajectories or live user simulator are used.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tau3_grpo.envs.adapter import airline_tool_schemas, build_environment, load_flight_db
from tau3_grpo.envs.generate_tool_config import _schema_for_verl
from tau3_grpo.envs.tau2_bridge import message_models, task_model
from tau3_grpo.evaluation.process_reward import DB_WRITE_TOOLS, score_turns
from tau3_grpo.paths import CODE_ROOT, DATA_ROOT
from tau3_grpo.utils.hashing import sha256_file

RECIPE = {"mode": "reference_write", "version": "v2"}


def record(call, k, error):
    return {"schema": "tau3_turn_v1", "turn_index": k, "token_span": [k, k + 1],
            "tool_calls": [{"name": call.name, "arguments": call.arguments, "error": error}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = CODE_ROOT / 'results/analysis/rl_curriculum50_20260912/manifests/areal_airline_train_seed42.jsonl'
    entries = {x['task_id']: x for x in map(json.loads, manifest.read_text().splitlines())}
    tasks = {k: task_model().model_validate(v['task']) for k, v in entries.items()}
    counts, potentials, details = Counter(), [], []
    tool_call_cls = message_models()['ToolCall']
    for task_id, task in tasks.items():
        basis = [b.value for b in task.evaluation_criteria.reward_basis]
        gold = [g.model_dump(mode='json') for g in task.evaluation_criteria.actions or []]
        counts['tasks'] += 1
        counts['reference_calls'] += len(gold)
        counts['reference_db_writes'] += sum(g['name'] in DB_WRITE_TOOLS for g in gold)
        counts['action_basis_tasks'] += 'ACTION' in basis
        env = build_environment(load_flight_db(DATA_ROOT / 'raw/areal_tau2' / entries[task_id]['db_path']))
        records = []
        for k, action in enumerate(gold):
            call = tool_call_cls(id=f'gold-{k}', name=action['name'], arguments=action['arguments'])
            response = env.get_response(call)
            records.append(record(call, k, bool(response.error)))
            counts['reference_execution_errors'] += bool(response.error)
        p = score_turns(records, gold, basis, RECIPE, official_outcome=1)
        positive = sum(max(e['reward'], 0) for t in p['turn_records'] for e in t['tool_calls'])
        assert positive <= 1 + 1e-12
        zero = score_turns(records, gold, basis, RECIPE, official_outcome=0)
        assert all(e['reward'] <= 0 for t in zero['turn_records'] for e in t['tool_calls'])
        potentials.append({'task_id': task_id, 'reference_write_count': p['reference_write_count'],
                           'positive_budget_if_official_success': positive})
    # Use the production argument parser, rather than an audit-specific coercion.
    from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser
    from verl.tools.schemas import OpenAIFunctionToolSchema
    xml = Qwen3XMLToolParser(None)
    schemas = [OpenAIFunctionToolSchema.model_validate(_schema_for_verl(x)) for x in airline_tool_schemas()]
    for path in args.input:
        if path.parent.name != 'rollouts':
            raise ValueError('Only historical training rollout files are accepted')
        for row in map(json.loads, path.read_text().splitlines()):
            task_id = row['task_id']
            if task_id not in tasks:
                raise ValueError('Non-training task in scope audit')
            counts['historical_rows'] += 1
            trajectory = json.loads(row['trajectory_json'])
            env = build_environment(load_flight_db(DATA_ROOT / 'raw/areal_tau2' / entries[task_id]['db_path']))
            item = {'source': str(path), 'row': counts['historical_rows'], 'task_id': task_id,
                    'official_outcome': row['score']}
            try:
                assert env.get_db_hash() == row['initial_db_hash'], 'initial_db_mismatch'
                calls = []
                for a, b in xml.tool_call_function_regex.findall(row['output']):
                    raw = xml._parse_xml_function_call(a or b, schemas)
                    calls.append(tool_call_cls(id=f'replay-{len(calls)}', name=raw.name,
                                               arguments=json.loads(raw.arguments)))
                assert len(calls) == trajectory['tool_calls'], 'call_count_mismatch'
                records = [record(call, k, bool(env.get_response(call).error)) for k, call in enumerate(calls)]
                assert env.get_db_hash() == row['db_hash'], 'final_db_mismatch'
                task = tasks[task_id]
                gold = [g.model_dump(mode='json') for g in task.evaluation_criteria.actions or []]
                basis = [b.value for b in task.evaluation_criteria.reward_basis]
                reward = score_turns(records, gold, basis, RECIPE, official_outcome=row['score'])
                events = [e for t in reward['turn_records'] for e in t['tool_calls']]
                positive = sum(max(e['reward'], 0) for e in events)
                assert positive <= 1 + 1e-12
                assert row['score'] == 1 or positive == 0
                assert all(e['name'] in DB_WRITE_TOOLS for e in events if e['reward'] > 0)
                counts['replayed_rows'] += 1
                counts['replayed_successes'] += row['score'] == 1
                counts['positive_reward_rows'] += positive > 0
                counts['positive_reward_calls'] += sum(e['reward'] > 0 for e in events)
                counts['error_penalty_calls'] += sum(e['reward'] < 0 for e in events)
                item.update(status='call_and_db_replayed', positive_reward=positive,
                            net_reward=sum(reward['turn_rewards']))
            except (AssertionError, ValueError, KeyError) as exc:
                counts['unverified_rows'] += 1
                item.update(status='unverified', reason=str(exc))
            details.append(item)
    report = {'recipe': RECIPE, 'counts': dict(counts), 'task_potentials': potentials, 'historical': details,
              'sources': {str(p): sha256_file(p) for p in [manifest, *args.input]},
              'limitations': ['Reference outcomes 1/0 are synthetic gate checks, not measured successes.',
                             'Historical replay verifies call count and DB endpoints, not every observation.',
                             'Synthetic call spans are not generation spans; no advantage replay claim.',
                             'No independent policy/authorization verifier; outcome gate inherits official limits.',
                             'No weight fitting or selection/final data used.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({'counts': dict(counts), 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
