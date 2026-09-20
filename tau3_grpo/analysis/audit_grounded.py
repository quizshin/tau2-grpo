"""Audit bounded field extraction on DB-certified, strictly pre-action prefixes."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import yaml

from tau3_grpo.algorithms.anchors.grounded import extract, validate
from tau3_grpo.algorithms.anchors.evidence import decision_evidence
from tau3_grpo.analysis.replay_decisions import as_object
from tau3_grpo.data.messages import visible_message
from tau3_grpo.utils.hashing import sha256_file, sha256_json


def run(replay_dirs, output, *, enabled=False):
    if not enabled:
        return {'enabled': False, 'models_called': False, 'gpu_used': False}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts, sources, cache, partitions = Counter(), {}, {}, {}
    with (output / 'decisions.jsonl').open('w') as stream:
        for directory in map(Path, replay_dirs):
            replay_summary = json.loads((directory / 'summary.json').read_text())
            sources[str(directory / 'summary.json')] = sha256_file(directory / 'summary.json')
            sources[str(directory / 'trajectories.jsonl')] = sha256_file(directory / 'trajectories.jsonl')
            for line in (directory / 'trajectories.jsonl').open():
                row = json.loads(line)
                if row['status'] != 'verified':
                    raise ValueError('Only fully verified trajectories are accepted')
                source = row['source']
                if source not in cache:
                    actual_hash = sha256_file(source)
                    if actual_hash != replay_summary['source_sha256'][source]:
                        raise ValueError('Source changed since strict DB replay')
                    sources[source] = actual_hash
                    cache[source] = [json.loads(line) for line in Path(source).open()]
                raw = cache[source][row['source_line'] - 1]
                if (raw['task_id'], raw['trial']) != (row['task_id'], row['trial']):
                    raise ValueError('Replay/source identity mismatch')
                messages = raw['simulation']['messages']
                counts['trajectories'] += 1
                for snapshot in row['snapshots']:
                    index = snapshot['message_index']
                    prefix = [visible_message(m) for m in messages[:index]]
                    if not snapshot['certified_prefix'] or sha256_json(prefix) != snapshot['prefix_hash']:
                        raise ValueError('Prefix mismatch')
                    report = extract(prefix)
                    knowledge = decision_evidence([as_object(m) for m in prefix], version='v2')
                    knowledge_guard = sha256_json([knowledge.read_hash, knowledge.tool_event_hash])
                    errors = validate(prefix, report)
                    if errors:
                        raise ValueError(f'Evidence verification failed: {errors}')
                    counts['decisions'] += 1
                    counts['evidence_validation_passed'] += 1
                    counts['with_claims'] += bool(report['claims'])
                    counts['with_observed_association'] += bool(report['decision_view']['observed_associations'])
                    counts['with_communicated_money'] += bool(report['decision_view']['communicated_quotes'])
                    counts['with_payment_correspondence'] += bool(report['decision_view']['payment_correspondences'])
                    counts['semantic_complete'] += report['semantic_complete']
                    counts['training_eligible'] += report['training_eligible']
                    for kind in set(c['kind'] for c in report['claims']):
                        counts['positions_with/' + kind] += 1
                    for issue in set(i['kind'] for i in report['issues']):
                        counts['positions_with_issue/' + issue] += 1
                    counts['ambiguous_payment_links'] += sum(link['status'] == 'ambiguous' for link in report['links'])
                    # Compact evidence receipt: full report can be rebuilt from
                    # pinned source+prefix. Do not duplicate large search results.
                    linked_facts = sorted(set(fi for link in report['links'] for fi in link['facts']))
                    receipt = {'task_id': row['task_id'], 'trial': row['trial'], 'source': source,
                               'message_index': index, 'db_hash': snapshot['db_hash'],
                               'policy_hash': snapshot['policy_hash'], 'report_sha256': sha256_json(report),
                               'prefix_sha256': report['prefix_sha256'], 'claims': report['claims'],
                               'linked_facts': {str(fi): report['facts'][fi] for fi in linked_facts},
                               'links': report['links'], 'issues': report['issues'],
                               'decision_view': report['decision_view'], 'guard_sha256': report['guard_sha256'],
                               'knowledge_guard_sha256': knowledge_guard,
                               'semantic_complete': False, 'training_eligible': False}
                    stream.write(json.dumps(receipt, ensure_ascii=False) + '\n')
                    # Full text guard is a diagnostic, never an enabled training key.
                    key = (source, row['task_id'], snapshot['db_hash'], snapshot['policy_hash'],
                           report['guard_sha256'], knowledge_guard)
                    partitions.setdefault(key, []).append((row['trial'], snapshot['assistant_step']))
    repeated = [xs for xs in partitions.values() if len({trial for trial, _ in xs}) > 1]
    counts['exact_guard_cross_trial_noninitial'] = sum(step > 0 for xs in repeated for _, step in xs)
    result = {'enabled': True, 'counts': dict(counts), 'source_sha256': sources,
              'extractor_sha256': sha256_file('tau3_grpo/algorithms/anchors/grounded.py'),
              'models_called': False, 'gpu_used': False,
              'limitations': ['Evidence validity is not semantic completeness or field recall.',
                             'Mentions do not establish intent, authorization, or a correct refund.',
                             'No training hook is connected; exact-guard repetition is not semantic grouping accuracy.']}
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--replay-dir', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if config.get('schema') != 'grounded_evidence_v1':
        raise ValueError('Unsupported extraction schema')
    result = run(args.replay_dir, args.output, enabled=config.get('enabled') is True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
