"""Explicit CPU-side API audit; credentials are loaded only when run is invoked."""
from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import ExitStack
from pathlib import Path

import yaml

from tau3_grpo.algorithms.anchors.semantic_state import SemanticError, compile_state
from tau3_grpo.models.semantic_api import OpenAICompatibleSemanticModel, SemanticAPIError
from tau3_grpo.models.semantic_extractor import PROMPT_PATH, build_request, slot_prompt_path
from tau3_grpo.utils.hashing import sha256_file


async def run(config, output, *, transport=None):
    if config.get('enabled') is not True:
        return {'enabled': False}
    if config.get('provider') != 'openai_compatible':
        raise ValueError('Expected openai_compatible provider')
    # Preflight before output creation or any network traffic.
    model = OpenAICompatibleSemanticModel.from_env(config, transport=transport)
    api_model = model
    dataset = json.loads(Path(config['cases']).read_text())
    cases = dataset['cases']
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case IDs')
    case_ids = config.get('case_ids')
    if case_ids is not None:
        if (not isinstance(case_ids, list) or not case_ids or len(set(case_ids)) != len(case_ids)
                or not set(case_ids) <= {c['id'] for c in cases}):
            raise ValueError('case_ids must select unique known cases')
        cases = [c for c in cases if c['id'] in case_ids]
    limit = config.get('limit', 2)
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError('limit must be a positive integer or null')
    cases = cases[:limit]
    concurrency = config.get('concurrency', 1)
    if type(concurrency) is not int or not 1 <= concurrency <= 4:
        raise ValueError('concurrency must be between 1 and 4')
    slot_schema = config.get('slot_schema')
    build_request([], slot_schema=slot_schema)  # Validate switch before network/output.
    mode = config.get('extraction_mode', 'full_prefix')
    if mode == 'incremental_v1':
        from tau3_grpo.models.semantic_incremental import IncrementalSemanticModel, PROMPT as INCREMENTAL_PROMPT
        saved, saved_provenance = None, None
        if config.get('resume_from'):
            source = Path(config['resume_from'])
            provenance = json.loads((source / 'summary.json').read_text())['provenance']
            expected = {'slot_schema': slot_schema, 'extraction_mode': mode,
                        'slot_prompt_sha256': sha256_file(slot_prompt_path(slot_schema)),
                        'prompt_sha256': sha256_file(PROMPT_PATH),
                        'incremental_prompt_sha256': sha256_file(INCREMENTAL_PROMPT),
                        'model': model.model, 'base_url': model.base_url,
                        'temperature': model.temperature, 'max_tokens': model.max_tokens,
                        'thinking_mode_requested': model.thinking_mode,
                        'cases_sha256': sha256_file(config['cases'])}
            if any(provenance.get(k) != v for k, v in expected.items()):
                raise ValueError('Resume source model/schema/prompt mismatch')
            source_rows = [json.loads(line) for line in (source / 'deltas.jsonl').read_text().splitlines()]
            saved = {r['prefix_sha256']: r for r in source_rows}
            if len(saved) != len(source_rows):
                raise ValueError('Duplicate resume prefix')
            saved_provenance = {'path': str(source), 'summary_sha256': sha256_file(source / 'summary.json'),
                                'deltas_sha256': sha256_file(source / 'deltas.jsonl')}
        model = IncrementalSemanticModel(model, slot_schema=slot_schema,
                                         max_calls=config.get('max_incremental_calls', 32),
                                         saved_deltas=saved, saved_provenance=saved_provenance,
                                         diagnostic_continue=config.get('diagnostic_continue', False))
    elif mode != 'full_prefix':
        raise ValueError('Unsupported extraction_mode')
    elif config.get('resume_from'):
        raise ValueError('resume_from requires incremental_v1')
    semaphore = asyncio.Semaphore(concurrency)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results, cached = {}, {}
    # Neither pair labels nor task metadata/returns are sent to the API.
    with ExitStack() as stack:
        raw_responses = stack.enter_context((output / 'raw_responses.jsonl').open('w'))

        def persist_raw_response(record):
            raw_responses.write(json.dumps(record, ensure_ascii=False) + '\n')
            raw_responses.flush()

        api_model.on_raw_response = persist_raw_response
        packets = stack.enter_context((output / 'packets.jsonl').open('w'))
        states = stack.enter_context((output / 'states.jsonl').open('w'))
        if mode == 'incremental_v1':
            deltas = stack.enter_context((output / 'deltas.jsonl').open('w'))

            def persist_delta(record):
                deltas.write(json.dumps(record, ensure_ascii=False) + '\n')
                deltas.flush()

            model.on_delta = persist_delta
        async def extract(request):
            async with semaphore:
                try:
                    packet, error = await model.extract(request), None
                except (SemanticError, SemanticAPIError) as exc:
                    packet, error = None, str(exc)
                packets.write(json.dumps({'prefix_sha256': request['prefix_sha256'], 'packet': packet,
                                         'raw_packet': model.response_packets.get(request['prefix_sha256']),
                                         'error': error},
                                         ensure_ascii=False) + '\n')
                packets.flush()
                return packet, error

        async def process(case):
            request = build_request(case['messages'], slot_schema=slot_schema)
            prefix = request['prefix_sha256']
            if prefix not in cached:
                cached[prefix] = asyncio.create_task(extract(request))
            packet, error = await cached[prefix]
            row = {'id': case['id'], 'key': None, 'error': error, 'simulated': False}
            row['response_available'] = prefix in model.response_packets
            if packet is not None:
                try:
                    compiled = compile_state(request['visible_messages'], packet,
                                             task_id=case['task_id'], db_hash=case['db_hash'],
                                             policy_hash=case['policy_hash'],
                                             remaining_turns=case.get('remaining_turns'),
                                             slot_schema=slot_schema)
                    row.update(key=compiled['key'], state=compiled['state'])
                except SemanticError as exc:
                    row['error'] = str(exc)
                except (TypeError, KeyError, IndexError, AttributeError):
                    row['error'] = 'Semantic state compilation rejected malformed field types'
            results[case['id']] = row
            states.write(json.dumps(row, ensure_ascii=False) + '\n')
            states.flush()
        await asyncio.gather(*(process(case) for case in cases))
    checks = []
    for pair in dataset['pairs']:
        if pair['a'] not in results or pair['b'] not in results:
            continue
        a, b = results[pair['a']]['key'], results[pair['b']]['key']
        actual = 'abstain' if a is None or b is None else 'merge' if a == b else 'separate'
        available = results[pair['a']]['response_available'] and results[pair['b']]['response_available']
        checks.append({**pair, 'actual': actual, 'response_available': available,
                       'passed': available and actual == pair['expected']})
    summary = {'simulated_model': False, 'api_request_attempts': model.attempted_calls,
               'gpu_used_locally': False, 'training_enabled': False,
               'cases': len(results), 'valid_cases': sum(r['key'] is not None for r in results.values()),
               'pairs': len(checks), 'passed_pairs': sum(c['passed'] for c in checks),
               'false_merges': sum(c['actual'] == 'merge' and c['expected'] == 'separate' for c in checks),
               'missed_merges': sum(c['actual'] == 'separate' and c['expected'] == 'merge' for c in checks),
               'abstained_pairs': sum(c['actual'] == 'abstain' for c in checks),
               'unavailable_pairs': sum(not c['response_available'] for c in checks),
               'checks': checks, 'usage': model.metadata, 'request_timings': model.request_timings,
               'provenance': {**model.provenance, 'cases_sha256': sha256_file(config['cases']),
                              'prompt_sha256': sha256_file(PROMPT_PATH), 'limit': limit,
                              'concurrency': concurrency, 'slot_schema': slot_schema,
                              'case_ids': case_ids,
                              'slot_prompt_sha256': sha256_file(slot_prompt_path(slot_schema))
                              if slot_schema is not None else None},
               'scope': 'Real-model extraction on supplied prefixes. Valid means protocol/evidence accepted; dataset provenance and semantic accuracy require a separate audit. Not evidence of RL improvement.'}
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
