"""Score frozen real-prefix labels and retrospective group signal, without fitting."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from tau3_grpo.utils.hashing import sha256_file, sha256_json


def score(cases, rows, labels, metadata):
    by_id = {c['id']: c for c in cases}
    results = {r['id']: r for r in rows}
    if len(by_id) != len(cases) or len(results) != len(rows) or set(by_id) != set(results):
        raise ValueError('Audit must contain exactly one result per frozen case')
    cohorts = defaultdict(list)
    for cid, case in by_id.items():
        meta = metadata[cid]
        if meta['prefix_sha256'] != sha256_json(case['messages']) or meta['task_id'] != case['task_id']:
            raise ValueError('Outcome identity/prefix mismatch')
        cohorts[(meta['source'], meta['task_id'], meta['assistant_step'])].append(cid)
    checks = []
    for pair in labels['pairs']:
        if pair['expected'] not in ('merge', 'separate'):
            raise ValueError('Invalid expected pair relation')
        a, b = results[pair['a']]['key'], results[pair['b']]['key']
        actual = 'abstain' if a is None or b is None else 'merge' if a == b else 'separate'
        checks.append({**pair, 'actual': actual, 'passed': actual == pair['expected']})

    def metrics(items):
        return {'pairs': len(items), 'decidable': sum(p['actual'] != 'abstain' for p in items),
                'passed': sum(p['passed'] for p in items),
                'abstained': sum(p['actual'] == 'abstain' for p in items),
                'false_merges': sum(p['actual'] == 'merge' and p['expected'] == 'separate' for p in items),
                'missed_merges': sum(p['actual'] == 'separate' and p['expected'] == 'merge' for p in items)}

    groups = []
    for (source, task, step), ids in sorted(cohorts.items()):
        if len({metadata[cid]['trial'] for cid in ids}) != len(ids):
            raise ValueError('Duplicate trial in a task/position cohort')
        buckets = defaultdict(list)
        for cid in ids:
            if results[cid]['key'] is not None:
                buckets[results[cid]['key']].append(cid)
        for key, members in buckets.items():
            if len(members) < 2:
                continue
            rewards = [metadata[cid]['reward'] for cid in members]
            groups.append({'source': source, 'task': task, 'step': step, 'key': key,
                           'members': members, 'rewards': rewards,
                           'variable_return': len(set(rewards)) > 1,
                           'proper_subgroup': len(members) < len(ids),
                           'cohort_size': len(ids),
                           'all_cohort_states_available': all(results[cid]['key'] is not None for cid in ids)})
    by_step = {}
    for step in sorted({m['assistant_step'] for cid, m in metadata.items() if cid in by_id}):
        selected = [results[cid] for cid in by_id if metadata[cid]['assistant_step'] == step]
        by_step[str(step)] = {'cases': len(selected), 'valid': sum(r['key'] is not None for r in selected)}
    return {'cases': len(rows), 'valid_cases': sum(r['key'] is not None for r in rows),
            'unique_prefixes': len({metadata[cid]['prefix_sha256'] for cid in by_id}),
            'valid_unique_prefixes': len({metadata[cid]['prefix_sha256'] for cid in by_id if results[cid]['key'] is not None}),
            'errors': dict(Counter(r['error'] for r in rows if r['key'] is None)),
            'valid_with_opaque_context': sum(bool(r.get('state', {}).get('opaque_context')) for r in rows if r['key'] is not None),
            'by_step': by_step,
            'exact_prefix_checks': metrics([p for p in checks if p['category'] == 'exact_prefix']),
            'semantic_checks': metrics([p for p in checks if p['category'] != 'exact_prefix']),
            'by_category': {cat: metrics([p for p in checks if p['category'] == cat])
                            for cat in sorted({p['category'] for p in checks})},
            'checks': checks, 'comparable_groups': groups,
            'variable_return_groups': sum(g['variable_return'] for g in groups),
            'variable_return_proper_subgroups': sum(g['variable_return'] and g['proper_subgroup'] for g in groups),
            'fully_observed_variable_return_proper_subgroups': sum(
                g['variable_return'] and g['proper_subgroup'] and g['all_cohort_states_available'] for g in groups),
            'scope': 'Evaluation trials, aligned by task and sampled decision position; not original RL groups or an estimate of training gains. Valid means protocol/evidence accepted, not verified semantic accuracy.'}


def run(cases_path, states_path, labels_path, metadata_path, output):
    paths = [Path(p) for p in (cases_path, states_path, labels_path, metadata_path)]
    cases_path, states_path, labels_path, metadata_path = paths
    labels = json.loads(labels_path.read_text())
    if labels['cases_sha256'] != sha256_file(cases_path):
        raise ValueError('Review labels do not match frozen cases')
    result = score(json.loads(cases_path.read_text())['cases'],
                   [json.loads(line) for line in states_path.read_text().splitlines()],
                   labels, json.loads(metadata_path.read_text()))
    result['provenance'] = {str(p): sha256_file(p) for p in paths}
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser()
    for arg in ('cases', 'states', 'labels', 'metadata', 'output'):
        p.add_argument('--'+arg, required=True)
    args = p.parse_args()
    run(args.cases, args.states, args.labels, args.metadata, args.output)


if __name__ == '__main__':
    main()
