"""Revalidate saved API packets without further inference or training."""
import argparse
import json
from pathlib import Path

import yaml

from tau3_grpo.algorithms.anchors.semantic_state import SemanticError, compile_state
from tau3_grpo.models.semantic_extractor import build_request
from tau3_grpo.utils.hashing import sha256_file


def run(config, source, output, *, supplements=()):
    source, output = Path(source), Path(output)
    source_rows = [json.loads(line) for line in (source / 'packets.jsonl').read_text().splitlines()]
    packets = {row['prefix_sha256']: row for row in source_rows}
    if len(packets) != len(source_rows):
        raise ValueError('Duplicate saved prefix')
    source_hashes = [sha256_file(source / 'packets.jsonl')]
    for extra in supplements:
        path = Path(extra) / 'packets.jsonl'
        extra_rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len({r['prefix_sha256'] for r in extra_rows}) != len(extra_rows):
            raise ValueError('Duplicate supplement prefix')
        for row in extra_rows:
            prefix = row['prefix_sha256']
            prior = packets.get(prefix)
            if prior is None or prior.get('raw_packet') is not None or prior.get('packet') is not None:
                raise ValueError('Supplement may only replace an existing prefix with no model packet')
            packets[prefix] = row
        source_hashes.append(sha256_file(path))
    dataset = json.loads(Path(config['cases']).read_text())
    cases = dataset['cases'][:config.get('limit', 2)]
    results = {}
    for case in cases:
        request = build_request(case['messages'], slot_schema=config.get('slot_schema'))
        saved = packets.get(request['prefix_sha256'])
        row = {'id': case['id'], 'key': None, 'error': 'no_saved_packet', 'response_available': False}
        if saved is not None:
            packet = saved.get('raw_packet') or saved.get('packet')
            row['error'] = saved.get('error') or 'no_saved_packet'
            if packet is not None:
                row['response_available'] = True
                try:
                    state = compile_state(request['visible_messages'], packet, task_id=case['task_id'],
                                          db_hash=case['db_hash'], policy_hash=case['policy_hash'],
                                          remaining_turns=case.get('remaining_turns'),
                                          slot_schema=config.get('slot_schema'))
                    row.update(key=state['key'], state=state['state'], error=None)
                except SemanticError as exc:
                    row['error'] = str(exc)
                except (TypeError, KeyError, IndexError, AttributeError):
                    row['error'] = 'malformed_field_types'
        results[case['id']] = row
    checks = []
    for pair in dataset['pairs']:
        if pair['a'] not in results or pair['b'] not in results:
            continue
        a, b = results[pair['a']]['key'], results[pair['b']]['key']
        actual = 'abstain' if a is None or b is None else 'merge' if a == b else 'separate'
        available = results[pair['a']]['response_available'] and results[pair['b']]['response_available']
        checks.append({**pair, 'actual': actual, 'response_available': available,
                       'passed': available and actual == pair['expected']})
    root = Path(__file__).resolve().parents[2]
    summary = {'new_api_calls': 0, 'training_enabled': False,
               'cases': len(results), 'valid_cases': sum(r['key'] is not None for r in results.values()),
               'pairs': len(checks), 'passed_pairs': sum(c['passed'] for c in checks),
               'false_merges': sum(c['actual'] == 'merge' and c['expected'] == 'separate' for c in checks),
               'missed_merges': sum(c['actual'] == 'separate' and c['expected'] == 'merge' for c in checks),
               'abstained_pairs': sum(c['actual'] == 'abstain' for c in checks),
               'unavailable_pairs': sum(not c['response_available'] for c in checks),
               'checks': checks, 'errors': {k: r['error'] for k, r in results.items() if r['error']},
               'provenance': {'source_packets_sha256': sha256_file(source / 'packets.jsonl'),
                              'all_source_packets_sha256': source_hashes,
                              'cases_sha256': sha256_file(config['cases']),
                              'slot_schema': config.get('slot_schema'),
                              'slots_code_sha256': sha256_file(root / 'tau3_grpo/algorithms/anchors/semantic_slots.py'),
                              'state_code_sha256': sha256_file(root / 'tau3_grpo/algorithms/anchors/semantic_state.py')}}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'packets.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n'
                                               for r in packets.values()))
    (output / 'states.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in results.values()))
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--supplement', action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(run(yaml.safe_load(Path(args.config).read_text()), args.source, args.output,
                         supplements=args.supplement), indent=2))


if __name__ == '__main__':
    main()
