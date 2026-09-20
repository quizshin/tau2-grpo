"""Revalidate immutable saved deltas; never call a model or repair its output."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import yaml

from tau3_grpo.algorithms.anchors.semantic_state import SCHEMA, SemanticError, compile_state
from tau3_grpo.models.semantic_extractor import build_request
from tau3_grpo.models.semantic_incremental import append_delta, delta_request
from tau3_grpo.utils.hashing import sha256_file, sha256_json


def run(config, source, output):
    source, output = Path(source), Path(output)
    schema = config.get('slot_schema', 'airline_slots_v2')
    build_request([], slot_schema=schema)
    dataset = json.loads(Path(config['cases']).read_text())
    cases = dataset['cases']
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case IDs')
    selected = config.get('case_ids')
    if selected is not None:
        if (not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
                or not set(selected) <= {c['id'] for c in cases}):
            raise ValueError('case_ids must select unique known cases')
        cases = [c for c in cases if c['id'] in selected]
    limit = config.get('limit')
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError('limit must be a positive integer or null')
    cases = cases[:limit]
    rows = [json.loads(line) for line in (source / 'deltas.jsonl').read_text().splitlines()]
    saved = {row['prefix_sha256']: row for row in rows}
    if len(saved) != len(rows):
        raise ValueError('Duplicate saved delta prefix')
    cache = {sha256_json([]): ({'schema': SCHEMA, 'prefix_sha256': sha256_json([]), 'events': []}, None, None)}
    replayed, results = [], []

    def prefix(messages):
        key = sha256_json(messages)
        if key in cache:
            return deepcopy(cache[key])
        prior, error, failure_kind = prefix(messages[:-1])
        if prior is None:
            result = (None, error, 'upstream_blocked:' + failure_kind.removeprefix('upstream_blocked:'))
        else:
            current = messages[-1]
            row = saved.get(key)
            needs_delta = current.get('role') in ('user', 'assistant') and bool((current.get('content') or '').strip())
            if needs_delta and row is not None and row.get('diagnostic_only'):
                result = (None, 'diagnostic_delta_not_eligible', 'source_unavailable')
            elif needs_delta and (row is None or row.get('delta') is None):
                result = (None, 'no_saved_delta', 'source_unavailable')
            else:
                try:
                    if needs_delta:
                        if row['at'] != len(messages) - 1:
                            raise SemanticError('saved_delta_position_mismatch')
                        _, payload = delta_request(messages, prior, slot_schema=schema)
                        packet = append_delta(messages, prior, row['delta'], payload['evidence_catalog'], slot_schema=schema)
                    else:
                        packet = {**deepcopy(prior), 'prefix_sha256': key}
                        compile_state(messages, packet, task_id='replay', db_hash='local', policy_hash='local', slot_schema=schema)
                    result = (packet, None, None)
                except SemanticError as exc:
                    result = (None, str(exc), 'validation')
                except (KeyError, TypeError, IndexError, AttributeError):
                    result = (None, 'malformed_field_types', 'validation')
                if needs_delta:
                    replayed.append({'prefix_sha256': key, 'at': len(messages) - 1,
                                     'original_error': row.get('error'),
                                     'error': result[1], 'delta_sha256': sha256_json(row['delta'])})
        cache[key] = deepcopy(result)
        return result

    for case in cases:
        messages = build_request(case['messages'], slot_schema=schema)['visible_messages']
        packet, error, failure_kind = prefix(messages)
        row = {'id': case['id'], 'key': None, 'error': error, 'failure_kind': failure_kind,
               'response_available': packet is not None, 'simulated': False}
        if packet is not None:
            try:
                compiled = compile_state(messages, packet, task_id=case['task_id'], db_hash=case['db_hash'],
                                         policy_hash=case['policy_hash'], remaining_turns=case.get('remaining_turns'),
                                         slot_schema=schema)
                row.update(key=compiled['key'], state=compiled['state'])
            except SemanticError as exc:
                row.update(error=str(exc), failure_kind='validation')
        results.append(row)
    root = Path(__file__).resolve().parents[2]
    code_paths = ['algorithms/anchors/semantic_state.py', 'algorithms/anchors/semantic_slots.py',
                  'algorithms/anchors/semantic_slots_v2.py', 'algorithms/anchors/semantic_slots_v3.py',
                  'models/semantic_extractor.py', 'models/semantic_incremental.py',
                  'analysis/replay_semantic_incremental.py']
    if schema == 'airline_slots_v4':
        code_paths += ['algorithms/anchors/semantic_slots_v4.py',
                       'algorithms/anchors/semantic_questions.py']
    summary = {'new_api_calls': 0, 'training_enabled': False, 'raw_deltas_modified': False,
               'cases': len(results), 'valid_cases': sum(r['key'] is not None for r in results),
               'errors': dict(Counter(r['error'] for r in results if r['error'])),
               'failure_kinds': dict(Counter(r['failure_kind'] for r in results if r['failure_kind'])),
               'replayed_deltas': len(replayed),
               'provenance': {'source_deltas_sha256': sha256_file(source / 'deltas.jsonl'),
                              'cases_sha256': sha256_file(config['cases']), 'slot_schema': schema,
                              'case_ids': selected, 'limit': limit,
                              'code_sha256': {p: sha256_file(root / 'tau3_grpo' / p) for p in code_paths}},
               'scope': 'Revalidation of saved model outputs, not fresh extraction accuracy or an independent holdout. Missing suffix deltas remain unavailable; no outputs are invented.'}
    output.mkdir(parents=True, exist_ok=False)
    for name, items in [('states', results), ('replayed_deltas', replayed)]:
        (output / (name + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in items))
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser()
    for argument in ('config', 'source', 'output'):
        parser.add_argument('--' + argument, required=True)
    args = parser.parse_args()
    print(json.dumps(run(yaml.safe_load(Path(args.config).read_text()), args.source, args.output), indent=2))


if __name__ == '__main__':
    main()
