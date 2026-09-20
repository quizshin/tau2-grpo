"""Score frozen pair judgments; keep filters, uncertain labels and semantic tests separate."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path

from tau3_grpo.analysis.audit_semantic_pairs import clique_verified
from tau3_grpo.utils.hashing import sha256_file


def metrics(rows):
    labeled = [r for r in rows if r['expected'] in ('merge', 'separate')]
    return {'total': len(rows), 'labeled': len(labeled), 'unscored': len(rows)-len(labeled),
            'decidable': sum(r['actual'] != 'abstain' for r in labeled),
            'correct': sum(r['actual'] == r['expected'] for r in labeled),
            'abstained': sum(r['actual'] == 'abstain' for r in labeled),
            'false_merges': sum(r['actual'] == 'merge' and r['expected'] == 'separate' for r in labeled),
            'missed_merges': sum(r['actual'] == 'separate' and r['expected'] == 'merge' for r in labeled)}


def score(labels, rows, cases):
    by_id = {r['id']: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != {p['id'] for p in labels['pairs']}:
        raise ValueError('exactly_one_result_per_pair_required')
    checks, cohorts, relations = [], defaultdict(set), {}
    for label in labels['pairs']:
        row = by_id[label['id']]
        if (row['a'], row['b']) != (label['a'], label['b']):
            raise ValueError('result_pair_identity_mismatch')
        if label['expected'] not in ('merge', 'separate', 'unresolved') or row['actual'] not in ('merge', 'separate', 'abstain'):
            raise ValueError('invalid_relation')
        a, b = cases[row['a']], cases[row['b']]
        if a['comparison_group'] != b['comparison_group']:
            raise ValueError('cross_cohort_scoring_unsupported')
        checks.append({**label, 'actual': row['actual'], 'stage': row['stage'],
                       'filter_reason': row['filter_reason']})
        cohorts[a['comparison_group']].update((row['a'], row['b']))
        relations[frozenset((row['a'], row['b']))] = row['actual']
    cliques, nontransitive = [], []
    for cohort, ids in sorted(cohorts.items()):
        ids = sorted(ids)
        if len(ids) > 8:
            raise ValueError('offline_clique_audit_limited_to_eight_members')
        valid = [set(subset) for n in range(2, len(ids)+1) for subset in combinations(ids, n)
                 if clique_verified(list(subset), relations)]
        for group in valid:
            if not any(group < other for other in valid):
                cliques.append({'cohort': cohort, 'members': sorted(group), 'all_edges_symmetric_merge': True,
                                'contains_unresolved_gold_edge': any(
                                    p['expected']=='unresolved' and {p['a'],p['b']} <= group for p in checks)})
        for triple in combinations(ids, 3):
            edges = [relations.get(frozenset(edge)) for edge in combinations(triple, 2)]
            if edges.count('merge')==2 and 'separate' in edges:
                nontransitive.append({'cohort': cohort, 'members': list(triple), 'relations': edges})
    requests = [r[o] for r in rows for o in ('forward','reverse') if r.get(o)]
    model_rows = [r for r in rows if r['stage']=='model']
    return {'overall': metrics(checks),
            'local_filter': metrics([c for c in checks if c['stage']=='local_filter']),
            'transport_invariance': metrics([c for c in checks if c['category']=='transport_id_invariance']),
            'substantive_model': metrics([c for c in checks if c['stage']=='model' and c['category']!='transport_id_invariance']),
            'by_category': {cat: metrics([c for c in checks if c['category']==cat]) for cat in sorted({c['category'] for c in checks})},
            'by_expected': {v:metrics([c for c in checks if c['expected']==v]) for v in ('merge','separate','unresolved')},
            'orientation_disagreements': sum(r['forward']['verdict']!=r['reverse']['verdict'] and not r['forward']['error'] and not r['reverse']['error'] for r in model_rows),
            'request_errors': dict(Counter(r['error'] for r in requests if r['error'])),
            'checks': checks, 'candidate_maximal_cliques': cliques, 'nontransitive_triangles': nontransitive,
            'scope': 'Not training groups or improvement estimates. Unresolved gold excluded before inference; filters and transport checks do not prove semantic model generalization. Cliques are candidates, not independently validated equivalence classes.'}


def run(cases_path, labels_path, output_dir, score_path):
    cases_path, labels_path, output_dir = map(Path, (cases_path, labels_path, output_dir))
    labels = json.loads(labels_path.read_text())
    if labels['cases_sha256'] != sha256_file(cases_path):
        raise ValueError('labels_case_hash_mismatch')
    start = json.loads((output_dir/'start.json').read_text())
    if start['cases_sha256'] != sha256_file(cases_path):
        raise ValueError('inference_case_hash_mismatch')
    cases = {c['id']: c for c in json.loads(cases_path.read_text())['cases']}
    rows = [json.loads(line) for line in (output_dir/'pairs.jsonl').read_text().splitlines()]
    result = score(labels, rows, cases)
    summary = json.loads((output_dir/'summary.json').read_text())
    result['calls'] = {'attempts': summary['requests'], 'finish_reasons': dict(Counter(x['finish_reason'] for x in summary['metadata'])),
                       'reported_total_tokens': sum(x['usage'].get('total_tokens',0) for x in summary['metadata']),
                       'summed_request_seconds': round(sum(x['elapsed_seconds'] for x in summary['request_timings']),3)}
    result['provenance'] = {str(p):sha256_file(p) for p in [cases_path,labels_path,output_dir/'pairs.jsonl',output_dir/'summary.json',Path(__file__)]}
    with Path(score_path).open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ('cases','labels','output-dir','score'):p.add_argument('--'+name,required=True)
    a=p.parse_args();s=run(a.cases,a.labels,a.output_dir,a.score)
    print(json.dumps({k:s[k] for k in ('overall','local_filter','substantive_model','orientation_disagreements','request_errors','calls')},indent=2))


if __name__=='__main__':main()
