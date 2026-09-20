"""Bounded, evidence-checked pair judgments; never invoked by an RL hook."""
from __future__ import annotations

import argparse
import asyncio
import json
from itertools import combinations
from pathlib import Path

from tau3_grpo.data.messages import visible_message
from tau3_grpo.algorithms.anchors.evidence import decision_evidence
from tau3_grpo.models.semantic_api import OpenAICompatibleSemanticModel, SemanticAPIError
from tau3_grpo.utils.hashing import sha256_file, sha256_json

PROMPT_PATH = Path(__file__).resolve().parents[2] / 'configs/prompts/semantic_pair_direct_v1.txt'


def make_request(a, b):
    histories = {side: [visible_message(m) for m in case['messages']]
                 for side, case in [('A', a), ('B', b)]}
    return {'schema': 'semantic_pair_comparison_v1', 'system': PROMPT_PATH.read_text(),
            'prefix_sha256': sha256_json(histories), 'visible_messages': histories}


def evidence_payload(histories):
    """Assign local IDs to exact content spans; do not ask the model to copy text."""
    catalogue, tagged = {}, {}
    for side, messages in histories.items():
        tagged[side] = []
        for index, message in enumerate(messages):
            tagged_message = {k: v for k, v in message.items() if k != 'content'}
            content = message.get('content')
            if isinstance(content, str):
                segments, offset = [], 0
                for line in content.splitlines(keepends=True):
                    for start in range(0, len(line), 600):
                        quote = line[start:start+600]
                        ref = f'{side}:m{index}:s{len(segments)}'
                        catalogue[ref] = {'side': side, 'message_index': index,
                                          'start': offset+start, 'end': offset+start+len(quote),
                                          'quote': quote}
                        segments.append({'ref': ref, 'text': quote})
                    offset += len(line)
                tagged_message['content_segments'] = segments
            else:
                tagged_message['content'] = content
            tagged[side].append({'message_index': index, **tagged_message})
    return {'schema': 'semantic_pair_evidence_ids_v1', 'histories': tagged}, catalogue


def validate(result, histories):
    if not isinstance(result, dict) or set(result) != {'verdict', 'shared_state', 'differences', 'evidence'}:
        raise ValueError('invalid_pair_fields')
    if result['verdict'] not in ('merge', 'separate', 'abstain'):
        raise ValueError('invalid_pair_verdict')
    for field in ('shared_state', 'differences'):
        items = result[field]
        if (not isinstance(items, list) or len(items) > 12
                or any(not isinstance(x, str) or not x.strip() or len(x) > 2000 for x in items)):
            raise ValueError('invalid_pair_explanation')
    if result['verdict'] == 'merge' and (result['differences'] or not result['shared_state']):
        raise ValueError('merge_with_difference_or_no_shared_state')
    if result['verdict'] == 'separate' and not result['differences']:
        raise ValueError('separate_without_difference')
    evidence = result['evidence']
    if not isinstance(evidence, list) or len(evidence) > 16:
        raise ValueError('invalid_pair_evidence')
    _, catalogue = evidence_payload(histories)
    if any(not isinstance(ref, str) or ref not in catalogue
           or not catalogue[ref]['quote'].strip() for ref in evidence):
        raise ValueError('invalid_pair_citation')
    if len(set(evidence)) != len(evidence):
        raise ValueError('duplicate_pair_citation')
    if result['verdict'] != 'abstain' and {catalogue[ref]['side'] for ref in evidence} != {'A', 'B'}:
        raise ValueError('decidable_pair_requires_both_sides')
    return result


def prefilter(a, b):
    """Only same declared cohort/task, policy, DB and observed facts are candidates."""
    for field in ('comparison_group', 'task_id', 'policy_hash', 'db_hash'):
        if not isinstance(a.get(field), str) or not a[field] or not isinstance(b.get(field), str) or not b[field]:
            raise ValueError('missing_pair_scope:' + field)
        if a[field] != b[field]:
            return 'scope_difference:' + field
    ea = decision_evidence([visible_message(m) for m in a['messages']], version='v2')
    eb = decision_evidence([visible_message(m) for m in b['messages']], version='v2')
    if ea.read_hash != eb.read_hash or ea.tool_event_hash != eb.tool_event_hash:
        return 'observed_evidence_difference'
    return None


def combine(forward, reverse):
    if (forward.get('error') or reverse.get('error')
            or forward.get('verdict') != reverse.get('verdict')):
        return 'abstain'
    return forward['verdict']


def clique_verified(members, relations):
    """No transitive closure: every edge must have a symmetric merge decision."""
    return len(members) >= 2 and len(set(members)) == len(members) and all(
        relations.get(frozenset((a, b))) == 'merge' for a, b in combinations(members, 2))


def load_inputs(config, cases_path, pairs_path):
    dataset = json.loads(Path(cases_path).read_text())
    cases = {c['id']: c for c in dataset['cases']}
    if len(cases) != len(dataset['cases']):
        raise ValueError('duplicate_case_id')
    pair_data = json.loads(Path(pairs_path).read_text())
    if pair_data['cases_sha256'] != sha256_file(cases_path):
        raise ValueError('pair_input_hash_mismatch')
    if config.get('cohort_mode') != 'evaluation_trials':
        raise ValueError('only_explicit_evaluation_trials_supported_no_RL_integration')
    concurrency, budget = config.get('concurrency', 4), config.get('max_calls', 80)
    if type(concurrency) is not int or not 1 <= concurrency <= 4 or type(budget) is not int or budget <= 0:
        raise ValueError('invalid_pair_request_limits')
    seen, selected = set(), []
    for pair in pair_data['pairs']:
        if set(pair) != {'id', 'a', 'b'} or any(not isinstance(v, str) or not v for v in pair.values()):
            raise ValueError('pair_manifest_must_not_contain_labels')
        if pair['id'] in seen or pair['a'] not in cases or pair['b'] not in cases or pair['a'] == pair['b']:
            raise ValueError('duplicate_or_unknown_pair')
        seen.add(pair['id'])
        selected.append((pair, prefilter(cases[pair['a']], cases[pair['b']])))
    if not selected or sum(reason is None for _, reason in selected)*2 > budget:
        raise ValueError('empty_or_over_budget_pair_plan')
    return cases, selected


async def run(config, cases_path, pairs_path, output, *, transport=None):
    cases, selected = load_inputs(config, cases_path, pairs_path)
    model = OpenAICompatibleSemanticModel.from_env(config, transport=transport)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    provenance = {'cases_sha256': sha256_file(cases_path), 'pairs_sha256': sha256_file(pairs_path),
                  'prompt_sha256': sha256_file(PROMPT_PATH), 'config': config,
                  'sources': {str(p): sha256_file(p) for p in [Path(__file__),
                             Path(__file__).parents[1] / 'models/semantic_api.py']},
                  'provider': model.provenance}
    (output/'start.json').write_text(json.dumps(provenance, indent=2)+'\n')
    semaphore = asyncio.Semaphore(config.get('concurrency', 4))
    rows = []
    with (output/'raw_responses.jsonl').open('w') as raw_file, (output/'requests.jsonl').open('w') as requests, (output/'pairs.jsonl').open('w') as results:
        def save_raw(raw):
            raw_file.write(json.dumps(raw, ensure_ascii=False)+'\n'); raw_file.flush()
        model.on_raw_response = save_raw

        async def judge(pair, reverse):
            a, b = cases[pair['a']], cases[pair['b']]
            request = make_request(b, a) if reverse else make_request(a, b)
            row = {'pair_id': pair['id'], 'orientation': 'BA' if reverse else 'AB',
                   'prefix_sha256': request['prefix_sha256'], 'verdict': 'abstain',
                   'result': None, 'error': None}
            async with semaphore:
                try:
                    payload, catalogue = evidence_payload(request['visible_messages'])
                    result = await model.extract_json(request, user_payload=payload)
                    row['result'] = validate(result, request['visible_messages'])
                    row['verdict'] = result['verdict']
                    row['resolved_evidence'] = [catalogue[ref] for ref in result['evidence']]
                except (SemanticAPIError, ValueError, TypeError, KeyError, IndexError) as exc:
                    row['error'] = str(exc)
                requests.write(json.dumps(row, ensure_ascii=False)+'\n'); requests.flush()
            return row

        async def process(pair, reason):
            row = {**pair, 'filter_reason': reason}
            if reason is not None:
                row.update(actual='separate', stage='local_filter', forward=None, reverse=None)
            else:
                f, b = await asyncio.gather(judge(pair, False), judge(pair, True))
                row.update(actual=combine(f, b), stage='model', forward=f, reverse=b)
            results.write(json.dumps(row, ensure_ascii=False)+'\n'); results.flush()
            rows.append(row)
        await asyncio.gather(*(process(pair, reason) for pair, reason in selected))
    summary = {'pairs': len(rows), 'requests': model.attempted_calls,
               'local_filtered': sum(r['stage']=='local_filter' for r in rows),
               'verdicts': {v: sum(r['actual']==v for r in rows) for v in ('merge','separate','abstain')},
               'metadata': model.metadata, 'request_timings': model.request_timings, 'provenance': provenance,
               'scope': 'Offline evaluation-trial pairs, not actual RL rollout groups. Evidence validation does not prove semantic accuracy. No labels supplied to inference; no retries.'}
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    return summary


def main():
    p = argparse.ArgumentParser()
    for arg in ('config', 'cases', 'pairs', 'output'):
        p.add_argument('--'+arg, required=True)
    a = p.parse_args()
    s = asyncio.run(run(json.loads(Path(a.config).read_text()), a.cases, a.pairs, a.output))
    print(json.dumps(s, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
