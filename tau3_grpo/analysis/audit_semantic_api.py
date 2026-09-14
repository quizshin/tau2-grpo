"""Explicit CPU-side API audit; credentials are loaded only when run is invoked."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from tau3_grpo.algorithms.anchors.semantic_state import SemanticError, compile_state
from tau3_grpo.models.semantic_api import OpenAICompatibleSemanticModel, SemanticAPIError
from tau3_grpo.models.semantic_extractor import PROMPT_PATH, build_request
from tau3_grpo.utils.hashing import sha256_file


async def run(config, output, *, transport=None):
    if config.get('enabled') is not True:
        return {'enabled': False}
    if config.get('provider') != 'openai_compatible':
        raise ValueError('Expected openai_compatible provider')
    # Preflight before output creation or any network traffic.
    model = OpenAICompatibleSemanticModel.from_env(config, transport=transport)
    dataset = json.loads(Path(config['cases']).read_text())
    cases = dataset['cases']
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case IDs')
    limit = config.get('limit', 2)
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError('limit must be a positive integer or null')
    cases = cases[:limit]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results, cached = {}, {}
    # Neither pair labels nor task metadata/returns are sent to the API.
    with (output / 'packets.jsonl').open('w') as packets, (output / 'states.jsonl').open('w') as states:
        for case in cases:
            request = build_request(case['messages'])
            prefix = request['prefix_sha256']
            if prefix not in cached:
                try:
                    cached[prefix] = (await model.extract(request), None)
                except (SemanticError, SemanticAPIError) as exc:
                    cached[prefix] = (None, str(exc))
                packet, error = cached[prefix]
                packets.write(json.dumps({'prefix_sha256': prefix, 'packet': packet, 'error': error},
                                         ensure_ascii=False) + '\n')
                packets.flush()
            packet, error = cached[prefix]
            row = {'id': case['id'], 'key': None, 'error': error, 'simulated': False}
            if packet is not None:
                try:
                    compiled = compile_state(request['visible_messages'], packet,
                                             task_id=case['task_id'], db_hash=case['db_hash'],
                                             policy_hash=case['policy_hash'],
                                             remaining_turns=case.get('remaining_turns'))
                    row.update(key=compiled['key'], state=compiled['state'])
                except SemanticError as exc:
                    row['error'] = str(exc)
                except (TypeError, KeyError, IndexError, AttributeError):
                    row['error'] = 'Semantic state compilation rejected malformed field types'
            results[case['id']] = row
            states.write(json.dumps(row, ensure_ascii=False) + '\n')
            states.flush()
    checks = []
    for pair in dataset['pairs']:
        if pair['a'] not in results or pair['b'] not in results:
            continue
        a, b = results[pair['a']]['key'], results[pair['b']]['key']
        actual = 'abstain' if a is None or b is None else 'merge' if a == b else 'separate'
        checks.append({**pair, 'actual': actual, 'passed': actual == pair['expected']})
    summary = {'simulated_model': False, 'api_request_attempts': model.attempted_calls,
               'gpu_used_locally': False, 'training_enabled': False,
               'cases': len(results), 'valid_cases': sum(r['key'] is not None for r in results.values()),
               'pairs': len(checks), 'passed_pairs': sum(c['passed'] for c in checks),
               'checks': checks, 'usage': model.metadata,
               'provenance': {**model.provenance, 'cases_sha256': sha256_file(config['cases']),
                              'prompt_sha256': sha256_file(PROMPT_PATH), 'limit': limit},
               'scope': 'Real-model extraction on authored contract cases; not real rollout accuracy or RL improvement.'}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = asyncio.run(run(yaml.safe_load(Path(args.config).read_text()), args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
