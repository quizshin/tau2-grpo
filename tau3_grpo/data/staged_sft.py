"""CPU preparation of complete-dialogue SFT candidates with explicit source targets.

This stage writes candidates, never claims judge acceptance or official task replay.
Held-out task content is used only by lexical leakage screening, never in samples.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from tau3_grpo.data.sft import load_complete_airline_dialogues, normalize_reason
from tau3_grpo.data.sft_expansion import SOURCE_SHA256, audit_tool_calls, coverage
from tau3_grpo.utils.hashing import sha256_file, sha256_json


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def near_duplicate(left, right, threshold=0.88):
    a, b = normalize_reason(left), normalize_reason(right)
    if not a or not b:
        return False
    sa, sb = set(a.split()), set(b.split())
    if len(sa & sb) / len(sa | sb) >= threshold:
        return True
    matcher = SequenceMatcher(None, a, b)
    return (matcher.real_quick_ratio() >= threshold and matcher.quick_ratio() >= threshold
            and matcher.ratio() >= threshold)


def ordered_tool_receipts(record):
    """Sequential batches must pair by call ID when present, otherwise in order."""
    pending, errors = [], []
    seen = set()
    for i, message in enumerate(record['messages']):
        if message['role'] == 'tool':
            if not pending:
                errors.append(f'{i}:orphan_tool')
                continue
            call = pending.pop(0)
            name = call.get('function', call).get('name')
            if message.get('name') != name:
                errors.append(f'{i}:out_of_order_tool')
            if call.get('id') and message.get('tool_call_id') != call['id']:
                errors.append(f'{i}:tool_id_mismatch')
        else:
            if pending:
                errors.append(f'{i}:missing_tool_receipt')
                pending = []
            for call in message.get('tool_calls') or []:
                if call.get('id'):
                    if call['id'] in seen:
                        errors.append(f'{i}:duplicate_tool_id')
                    seen.add(call['id'])
                pending.append(call)
    if pending:
        errors.append('unfinished_tools')
    return errors


def prepare(args):
    import yaml
    from transformers import AutoTokenizer

    from tau3_grpo.data.prepare_sft import _official_reasons
    from tau3_grpo.training.sft.dataset import build_supervised_example

    if args.output.exists():
        raise FileExistsError('Immutable candidate directory already exists')
    if sha256_file(args.source) != SOURCE_SHA256:
        raise ValueError('Pinned source changed')
    inventory = json.loads(args.inventory.read_text())
    if (inventory['summary']['source_sha256'] != SOURCE_SHA256
            or inventory['summary']['prefix_mismatch_rows']
            or inventory['summary']['answer_mismatch_rows']):
        raise ValueError('Source provenance audit missing or inconsistent')
    census = {r['source_dialog_id']: r for r in inventory['records']}
    frozen = json.loads(Path('configs/data/clean14_100_frozen_v1.json').read_text())
    historical = read_rows('data/sft/airline_sft_train_seed42.jsonl')
    old_dev = read_rows('data/sft/airline_sft_validation_seed42.jsonl')
    trained = set(frozen['selected_areal_ids']) | {r['metadata']['source_dialog_id'] for r in historical}
    heldout_ids = {r['metadata']['source_dialog_id'] for r in old_dev}
    # Conservatively protect all reserve tasks until a smaller confirmation subset is frozen.
    manifest_paths = [Path(f'data/manifests/areal_airline_{split}_seed42.jsonl')
                      for split in ('selection', 'reserve')]
    blocked = [r['metadata']['reason_for_call'] for r in old_dev] + _official_reasons()
    for path in manifest_paths:
        blocked.extend(r['task']['user_scenario']['instructions']['reason_for_call'] for r in read_rows(path))
    schemas = [e['tool_schema'] for e in yaml.safe_load(args.tools.read_text())['tools']]
    dialogues, stats = load_complete_airline_dialogues(args.source)
    eligible, rejected = [], []
    for dialogue in dialogues:
        sid = dialogue.source_dialog_id
        row = dialogue.to_record()
        issues = []
        if sid in heldout_ids:
            issues.append('historical_offline_dev')
        if not census[sid]['source_eligible']:
            issues.append('source_label_not_unanimous_1_1')
        issues += audit_tool_calls(row, schemas) + ordered_tool_receipts(row)
        if not dialogue.reason_for_call:
            issues.append('missing_intent')
        if not issues and any(near_duplicate(dialogue.reason_for_call, reason) for reason in blocked):
            issues.append('heldout_lexical_overlap')
        if issues:
            rejected.append({'id': sid, 'issues': issues})
            continue
        # Drop private reasoning; retain every visible message including unsupervised greetings.
        row['messages'] = [{k: v for k, v in m.items()
                            if k in ('role', 'content', 'tool_calls', 'tool_call_id', 'name')}
                           for m in row['messages']]
        row['supervision'] = {'version': 'approved_assistant_v1',
                              'message_indices': census[sid]['provided_target_message_indices'],
                              'basis': 'source_answer_positions_pending_quality_review'}
        row['metadata'].update(evidence_level='source_only', status='candidate_not_training_approved',
                               source_labels=census[sid]['source_labels'],
                               missing_turn_indices=census[sid]['missing_turn_indices'],
                               seed_pattern_task_id=census[sid]['seed_pattern_task_id'])
        eligible.append(row)
    def rank(row):
        return sha256_json({'seed': 42, 'id': row['metadata']['source_dialog_id']})
    # Freeze dev before judge scores; protect all near duplicates of every dev dialogue.
    dev = []
    for row in sorted(eligible, key=rank):
        if row['metadata']['source_dialog_id'] in trained:
            continue
        if any(near_duplicate(row['metadata']['reason_for_call'], r['metadata']['reason_for_call'])
               for r in historical + [x for x in eligible if x['metadata']['source_dialog_id'] in trained] + dev):
            continue
        dev.append(row)
        if len(dev) == 60:
            break
    if len(dev) != 60:
        raise ValueError(f'Only {len(dev)} independent never-trained dev dialogues available')
    train_pool = [r for r in eligible if not any(near_duplicate(r['metadata']['reason_for_call'],
                   d['metadata']['reason_for_call']) for d in dev)]
    frequencies = Counter(coverage(train_pool))
    def priority(row):
        names = set(census[row['metadata']['source_dialog_id']]['tool_counts'])
        rare = any(frequencies[name] <= 15 for name in names)
        # A: emphasize simpler source patterns, but preserve legitimate rare-tool candidates.
        return (not rare, census[row['metadata']['source_dialog_id']]['tool_calls'] > 20, rank(row))
    candidates = []
    for row in sorted(train_pool, key=priority):
        if any(near_duplicate(row['metadata']['reason_for_call'], r['metadata']['reason_for_call'])
               for r in candidates):
            continue
        candidates.append(row)
        if len(candidates) == 180:
            break
    args.output.mkdir(parents=True)
    write_rows(args.output / 'source_eligible_pool.jsonl', eligible)
    manifest = dict(status='frozen_before_judge', source_sha256=SOURCE_SHA256,
                    inventory_sha256=sha256_file(args.inventory), source_counts=vars(stats),
                    blocked_manifest_hashes={str(p): sha256_file(p) for p in manifest_paths},
                    official_reasons_sha256=sha256_json(_official_reasons()),
                    protected_reserve_tasks=888, lexical_threshold=0.88,
                    limitations=['Lexical screening is not proof of semantic independence.',
                                 'Original source task/DB linkage is unavailable; no official replay.',
                                 'Candidates are not accepted training records.'],
                    eligible=len(eligible), train_pool=len(train_pool),
                    dev_ids=[r['metadata']['source_dialog_id'] for r in dev],
                    candidate_ids=[r['metadata']['source_dialog_id'] for r in candidates],
                    rejected=rejected)
    (args.output / 'split_manifest.json').write_text(json.dumps(manifest, indent=2))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    for name, rows in [('candidates', candidates), ('offline_dev', dev)]:
        accepted, failures = [], []
        for row in rows:
            try:
                result = build_supervised_example(row['messages'], tokenizer, tools=schemas,
                         max_length=24576, approved_indices=row['supervision']['message_indices'])
                row['metadata'].update(rendered_tokens=result['n_total_tokens'], label_tokens=result['n_label_tokens'])
                accepted.append(row)
            except ValueError as exc:
                failures.append({'id': row['metadata']['source_dialog_id'], 'error': str(exc)})
        write_rows(args.output / f'{name}.jsonl', accepted)
        manifest[name] = dict(count=len(accepted), failures=failures, coverage=coverage(accepted),
                              total_tokens=sum(r['metadata']['rendered_tokens'] for r in accepted),
                              label_tokens=sum(r['metadata']['label_tokens'] for r in accepted))
        print(json.dumps({name: manifest[name]}), flush=True)
    manifest['files'] = {p.name: sha256_file(p) for p in args.output.glob('*.jsonl')}
    manifest['tokenizer_sha256'] = {p.name: sha256_file(p) for p in args.model.glob('*')
                                     if p.name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')}
    manifest['status'] = 'cpu_candidates_prepared_not_training_approved'
    (args.output / 'split_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('rejected','dev_ids','candidate_ids')}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('data/raw/areal_tau2/tau2_sft_train.jsonl'))
    p.add_argument('--inventory', type=Path, default=Path('results/analysis/areal_sft_strategy_20260925/source_inventory.json'))
    p.add_argument('--tools', type=Path, default=Path('configs/envs/tool_config.yaml'))
    p.add_argument('--model', type=Path, default=Path('models/Qwen3.5-4B'))
    p.add_argument('--output', type=Path, required=True)
    prepare(p.parse_args())


if __name__ == '__main__':
    main()
