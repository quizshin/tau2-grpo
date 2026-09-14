"""Exercise an assumed semantic model and real reducer; no network/GPU calls."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from tau3_grpo.algorithms.anchors.semantic_state import compile_state, scope_candidate_key, SemanticError
from tau3_grpo.algorithms.tau_gigpo import compute_tau_gigpo_advantage, steps_from_anchor_payload
from tau3_grpo.models.semantic_extractor import FixtureSemanticModel, build_request, PROMPT_PATH
from tau3_grpo.utils.hashing import sha256_file


async def run(config, output):
    if config.get('enabled') is not True:
        return {'enabled': False}
    if config.get('provider') != 'fixture':
        raise ValueError('Only simulated fixture provider is enabled in this release')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    model = FixtureSemanticModel(config['responses'])
    dataset = json.loads(Path(config['cases']).read_text())
    if len({case['id'] for case in dataset['cases']}) != len(dataset['cases']):
        raise ValueError('Duplicate case IDs')
    results = {}
    # Process all cases before loading/comparing expected pair relations.
    for case in dataset['cases']:
        request = build_request(case['messages'])
        try:
            packet = await model.extract(request)
            compiled = compile_state(request['visible_messages'], packet, task_id=case['task_id'],
                                     db_hash=case['db_hash'], policy_hash=case['policy_hash'],
                                     remaining_turns=case.get('remaining_turns'))
            results[case['id']] = {'id': case['id'], 'key': compiled['key'], 'state': compiled['state'],
                                   'error': None, 'simulated': True}
        except (SemanticError, LookupError) as exc:
            results[case['id']] = {'id': case['id'], 'key': None, 'error': str(exc), 'simulated': True}
    checks = []
    for pair in dataset['pairs']:
        a, b = results[pair['a']], results[pair['b']]
        actual = 'abstain' if a['key'] is None or b['key'] is None else 'merge' if a['key'] == b['key'] else 'separate'
        checks.append({**pair, 'actual': actual, 'passed': actual == pair['expected']})
    probes = []
    for probe in config.get('synthetic_credit_probes', []):
        ids, groups, rewards = probe['cases'], probe['episode_groups'], probe['synthetic_returns']
        if not len(ids) == len(groups) == len(rewards) or any(results[i]['key'] is None for i in ids):
            raise ValueError('Invalid synthetic credit probe')
        keys = [scope_candidate_key(results[i]['key'], group) for i, group in zip(ids, groups)]
        steps = steps_from_anchor_payload([[key] for key in keys], [[(0, 1)] for _ in keys])
        advantage, stats = compute_tau_gigpo_advantage(rewards, groups, steps, response_length=1)
        probes.append({'id': probe['id'], 'synthetic_returns': rewards, 'combined_advantages': advantage.tolist(),
                       'usable_anchor_groups': stats.usable_anchor_groups, 'synthetic_not_rl_measurement': True})
    result = {'simulated_model': True, 'real_model_calls': 0, 'gpu_used': False, 'training_enabled': False,
              'cases': len(results), 'valid_cases': sum(r['key'] is not None for r in results.values()),
              'pairs': len(checks), 'passed_pairs': sum(r['passed'] for r in checks), 'checks': checks,
              'synthetic_credit_probes': probes,
              'provenance': {'cases_sha256': sha256_file(config['cases']), 'responses_sha256': model.fingerprint,
                             'prompt_sha256': sha256_file(PROMPT_PATH)},
              'scope': 'State-transition/normalization contracts with pre-authored simulated semantic outputs; NOT real-model extraction accuracy.'}
    (output / 'states.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in results.values()))
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=asyncio.run(run(yaml.safe_load(Path(args.config).read_text()),args.output))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
