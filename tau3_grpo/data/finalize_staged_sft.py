"""Freeze an experimental Stage A set from completed, evidence-bound review records.

LLM review is a selection aid, not a claim of official replay or improved policy.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

from tau3_grpo.analysis.rubric_pilot import visible_events
from tau3_grpo.data.build_clean14 import _category
from tau3_grpo.data.sft_expansion import coverage, tool_names
from tau3_grpo.data.sft_policy_checks import audit_baggage_allowances
from tau3_grpo.data.staged_sft import near_duplicate, read_rows, write_rows
from tau3_grpo.utils.hashing import sha256_file, sha256_json, sha256_text


def sid(row):
    return row['metadata']['source_dialog_id']


def category(row):
    pattern = row['metadata'].get('seed_pattern_task_id')
    return _category(pattern) if pattern else 'supplemental'


def reviewed_pool(rows, review):
    accepted, held = [], []
    for row in rows:
        path = review / 'records' / f'{sid(row)}.json'
        if not path.exists():
            raise ValueError(f'Missing completed audit: {sid(row)}')
        verdict = json.loads(path.read_text())
        if verdict['visible_hash'] != sha256_json(visible_events(row['messages'])):
            raise ValueError(f'Review message identity mismatch: {sid(row)}')
        rules = audit_baggage_allowances(row['messages'])
        flags = []
        if verdict['status'] != 'review_ready':
            flags.append('judge_unscored')
        else:
            audit = verdict['audit']
            if (audit['recommendation'] != 'keep_candidate' or audit['issues']
                    or any(x['status'] in ('violated', 'unknown') for x in audit['requirements'])):
                flags.append('judge_requires_review_not_confirmed_bad_data')
        if any(x['status'] == 'violated' for x in rules):
            flags.append('deterministic_baggage_mismatch')
        if flags:
            held.append({'id': sid(row), 'flags': flags, 'baggage_checks': rules})
            continue
        row['metadata'].update(review_sha256=sha256_file(path),
            review_version='airline_rubric_v2', evidence_level='source_and_llm_reviewed_not_replay_verified',
            baggage_checks=rules, difficulty=verdict['audit']['difficulty'],
            status='experimental_sft_candidate')
        row['supervision']['basis'] += '_review_passed'
        accepted.append(row)
    return accepted, held


def choose_stage_a(pool, *, size=100):
    # Broad foundation mix; rare tools and source pattern quotas are soft targets.
    goals = {'single_pos': 30, 'single_neg': 15, 'double': 35, 'triple': 17, 'supplemental': 3}
    available = Counter(coverage(pool))
    tool_targets = {name: min(3, n) for name, n in available.items()}
    categories, seen_tools = Counter(), Counter()
    remaining, chosen = list(pool), []
    while remaining and len(chosen) < size:
        def priority(row):
            names = sorted(tool_names(row))
            # A legitimate first example of a tool has priority; then broad business distribution.
            unseen = sum(1 / available[n] for n in names if not seen_tools[n])
            deficit = sum(1 / available[n] for n in names if seen_tools[n] < tool_targets[n])
            cat = category(row)
            need = max(0, goals[cat] - categories[cat]) / goals[cat]
            return (-unseen, -need, -deficit,
                    sha256_json({'seed': 42, 'id': sid(row)}))
        row = min(remaining, key=priority)
        remaining.remove(row)
        if any(near_duplicate(row['metadata']['reason_for_call'], x['metadata']['reason_for_call'])
               for x in chosen):
            continue
        chosen.append(row)
        seen_tools.update(tool_names(row))
        categories[category(row)] += 1
    if len(chosen) != size:
        raise ValueError(f'Only {len(chosen)} accepted independent conversations, need {size}')
    return chosen, {'pattern_targets': goals, 'pattern_actual': dict(categories),
                    'tool_minimum_targets': tool_targets, 'coverage': dict(seen_tools)}


def finalize(args):
    import yaml
    from transformers import AutoTokenizer

    from tau3_grpo.training.sft.dataset import TrajectorySFTDataset

    if args.output.exists():
        raise FileExistsError('Never overwrite a frozen training set')
    summary = json.loads((args.review / 'summary.json').read_text())
    if summary['completed'] != summary['planned']:
        raise ValueError('Review batch incomplete')
    source = read_rows(args.candidates / 'candidates.jsonl')
    supplemental = read_rows(args.supplemental)
    dev = read_rows(args.candidates / 'offline_dev.jsonl')
    pool, train_held = reviewed_pool(source + supplemental, args.review)
    dev_pool, dev_held = reviewed_pool(dev, args.review)
    train, distribution = choose_stage_a(pool)
    if not dev_pool:
        raise ValueError('No clean offline validation examples')
    required = {e['tool_schema']['function']['name'] for e in
                yaml.safe_load(Path('configs/envs/tool_config.yaml').read_text())['tools']}
    if set(coverage(train)) != required:
        raise ValueError(f'Missing legitimate runtime coverage: {sorted(required - set(coverage(train)))}')
    if any(near_duplicate(t['metadata']['reason_for_call'], d['metadata']['reason_for_call'])
           for t in train for d in dev):
        raise ValueError('Train/dev near duplicate')
    for row in train + dev_pool:
        row['metadata']['source_dialogue_hash'] = row['metadata'].get('dialogue_hash')
        row['metadata']['dialogue_hash'] = sha256_text(json.dumps(row['messages'], sort_keys=True, separators=(',', ':')))
    final_output = args.output
    final_output.parent.mkdir(parents=True, exist_ok=True)
    args.output = Path(tempfile.mkdtemp(prefix=f'.{final_output.name}-', dir=final_output.parent))
    write_rows(args.output / 'train.jsonl', train)
    write_rows(args.output / 'validation.jsonl', dev_pool)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    schemas = [e['tool_schema'] for e in yaml.safe_load(Path('configs/envs/tool_config.yaml').read_text())['tools']]
    stats = {}
    for name, size in [('train', 100), ('validation', len(dev_pool))]:
        ds = TrajectorySFTDataset(args.output / f'{name}.jsonl', tokenizer, tools=schemas,
                                  max_length=24576, expected_size=size, require_approved_targets=True)
        stats[name] = ds.token_stats()
        ds.write_effective_jsonl(args.output / f'{name}_effective.jsonl')
    inventory = json.loads(Path('results/analysis/areal_sft_strategy_20260925/source_inventory.json').read_text())
    census = {r['source_dialog_id']:r for r in inventory['records']}
    report = dict(status='cpu_verified_experimental_dataset_gpu_not_started',
        train_count=100, validation_count=len(dev_pool), candidate_train_count=len(source)+len(supplemental),
        review_eligible_train=len(pool), held_train=train_held, held_validation=dev_held,
        train_ids=[sid(r) for r in train], validation_ids=[sid(r) for r in dev_pool],
        distribution=distribution, token_stats=stats,
        difficulty=dict(Counter(r['metadata']['difficulty']['level'] for r in train)),
        source_counts=dict(Counter(r['metadata']['source'] for r in train)),
        excluded_missing_source_assistant_positions=sum(len(census[sid(r)]['assistant_messages_without_source_target'])
                                                       for r in train if sid(r) in census),
        inputs={'candidate_manifest':sha256_file(args.candidates/'split_manifest.json'),
                'review_manifest':sha256_file(args.review/'manifest.json')},
        files={p.name:sha256_file(p) for p in args.output.glob('*.jsonl')},
        limitations=['Single-model review is not human gold or calibrated certification.',
                     'Held records are unresolved review candidates, not proven failures.',
                     'No official replay of source dialogues; supplements have prior execution receipts only.',
                     'Lexical split checks do not prove semantic independence.',
                     'Whole-dialogue native Qwen3.5 historical prefixes differ from live generation prefixes.',
                     'Offline loss cannot establish improvement over Base.'],
        proposed_training={'epochs':1,'microbatch':1,'accumulation':8,'optimizer_steps':13,'learning_rate':3e-5})
    (args.output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    if final_output.exists():
        raise FileExistsError(final_output)
    args.output.rename(final_output)
    print(json.dumps({k:v for k,v in report.items() if k not in ('held_train','held_validation','train_ids','validation_ids')},ensure_ascii=False),flush=True)


def visible_user_ids(row):
    """Protect IDs in both call arguments and returned profiles/reservations."""
    found = set()
    def walk(value):
        if isinstance(value, dict):
            if isinstance(value.get('user_id'), str):
                found.add(value['user_id'])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, str) and value.lstrip().startswith(('{', '[')):
            try:
                walk(json.loads(value))
            except ValueError:
                pass
    walk(row['messages'])
    if row.get('metadata', {}).get('entity_group'):
        found.add(row['metadata']['entity_group'])
    return found


def adjudicated_sources(rows, review_dir, decisions, protected_users):
    """Bind selection to the reviewed messages, then enforce held-out entities."""
    accepted, held = [], []
    index = {d['id']: d for d in decisions}
    if len(index) != len(decisions) or set(index) != {sid(r) for r in rows}:
        raise ValueError('Adjudication must cover the exact unique source cohort')
    for row in rows:
        decision = index[sid(row)]
        path = review_dir / 'records' / f'{sid(row)}.json'
        review = json.loads(path.read_text())
        digest = sha256_json(visible_events(row['messages']))
        if (digest != review['evidence_sha256'] or digest != decision['review_evidence_sha256']
                or sha256_file(path) != decision['review_record_sha256']):
            raise ValueError('Source review identity mismatch: ' + sid(row))
        conflict = visible_user_ids(row) & protected_users
        if decision['decision'] != 'include_experimental' or conflict:
            held.append(dict(id=sid(row),reason='heldout_entity' if conflict else 'unresolved_review',
                             entities=sorted(conflict)))
            continue
        row['metadata'].update(status='experimental_training_approved',
            evidence_level='source_static_two_reviews_and_adjudication_not_original_db_replayed',
            current_review_sha256=sha256_file(path), adjudication=decision,
            difficulty_vector_uncalibrated=review.get('review', {}).get('difficulty'))
        row['supervision']['basis'] = 'original_source_positions_after_evidence_bound_selection'
        accepted.append(row)
    return accepted, held


def finalize_repair(args):
    """Freeze original-source reuse plus executed, entity-split repair dialogues."""
    import math
    import yaml
    from transformers import AutoTokenizer
    from tau3_grpo.analysis.capability_distribution import sft_feature, summarize
    from tau3_grpo.data.sft_expansion import audit_tool_calls
    from tau3_grpo.data.staged_sft import ordered_tool_receipts
    from tau3_grpo.training.sft.dataset import TrajectorySFTDataset

    if args.output.exists():
        raise FileExistsError('Never overwrite a frozen repair set')
    cohort = args.repair_cohort
    plan = json.loads((cohort/'plan.json').read_text())
    receipt = json.loads((cohort/'summary.json').read_text())
    if receipt['errors'] or receipt['completed'] != receipt['planned']:
        raise ValueError('Grounded cohort has unresolved execution failures')
    if receipt['input_plan_sha256'] != sha256_file(cohort/'plan.json'):
        raise ValueError('Grounded plan changed')
    protected = set(plan['protected_users']) | set(plan['heldout_users'])
    source, held = adjudicated_sources(read_rows(args.candidates/'previously_reviewed_candidates.jsonl'),
        args.review, read_rows(args.source_decisions), protected)
    supplements = read_rows(cohort/'train.jsonl')
    probe = read_rows(cohort/'validation.jsonl')
    if len(supplements) != len(plan['train']) or len(probe) != len(plan['validation']):
        raise ValueError('Grounded split count changed')
    for row in supplements + probe:
        if row['metadata']['plan_sha256'] != sha256_file(cohort/'plan.json'):
            raise ValueError('Grounded row plan mismatch')
        row['metadata']['status'] = 'experimental_training_approved' if row in supplements else 'capability_dev_only'
        row['metadata']['semantic_review'] = 'procedural_branch_review_and_native_expected_state_receipt'
    train = source + supplements
    old_dev = read_rows('data/sft/staged_v2_A100_20260925/validation.jsonl')
    dev_users = set().union(*(visible_user_ids(r) for r in old_dev + probe))
    if any(visible_user_ids(r) & (protected | dev_users) for r in train):
        raise ValueError('Train/dev entity collision')
    identities = [sid(r) for r in train + old_dev + probe]
    if len(set(identities)) != len(identities):
        raise ValueError('Repeated source dialogue identity')
    schemas = [e['tool_schema'] for e in yaml.safe_load(Path('configs/envs/tool_config.yaml').read_text())['tools']]
    required = {s['function']['name'] for s in schemas}
    for row in train + probe:
        errors = audit_tool_calls(row, schemas) + ordered_tool_receipts(row)
        if errors:
            raise ValueError((sid(row), errors))
    if set(coverage(train)) != required:
        raise ValueError('Missing tool coverage in training')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f'.{args.output.name}-', dir=args.output.parent))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    stats, details, distributions = {}, {}, {}
    for split, rows in [('train', train), ('validation', old_dev), ('capability_dev', probe)]:
        write_rows(temp/f'{split}.jsonl', rows)
        dataset = TrajectorySFTDataset(temp/f'{split}.jsonl', tokenizer, tools=schemas,
            max_length=24576, expected_size=len(rows), require_approved_targets=True)
        stats[split] = dataset.token_stats()
        dataset.write_effective_jsonl(temp/f'{split}_effective.jsonl')
        details[split] = [dict(id=sid(r),source=r['metadata']['source'],
            total_tokens=e['n_total_tokens'],assistant_tokens=e['n_label_tokens'])
            for r,e in zip(rows,dataset.examples)]
        distributions[split] = summarize([sft_feature(r) for r in rows], 'observed_demonstrations_not_required_actions')
    stats['train_source_mix'] = {name:dict(dialogues=len(group),
        assistant_tokens=sum(d['assistant_tokens'] for d in group),
        total_tokens=sum(d['total_tokens'] for d in group)) for name in sorted({d['source'] for d in details['train']})
        for group in [[d for d in details['train'] if d['source']==name]]}
    for name, value in [('token_details', details), ('distributions', distributions),
                        ('protected_entities', dict(users=sorted(protected|dev_users), scope='protect_all_future_source_reuse'))]:
        (temp/f'{name}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    report = dict(status='cpu_verified_gpu_not_started',train_count=len(train),source_count=len(source),
        supplemental_count=len(supplements),validation_count=len(old_dev),capability_dev_count=len(probe),
        held_sources=held,token_stats=stats,tool_coverage=dict(coverage(train)),
        planned_optimizer_steps=math.ceil(len(train)/8),epochs=1,learning_rate=3e-5,
        source_decisions_sha256=sha256_file(args.source_decisions),cohort_plan_sha256=sha256_file(cohort/'plan.json'),
        source_file_sha256=sha256_file(__file__),files={p.name:sha256_file(p) for p in temp.glob('*.jsonl')},
        gpu_training_ready=True,live_selection_evaluation_ready=False,
        limitations=['Original source DB/task mapping not recovered; original dialogues not replay-certified.',
            'New dev is entity-disjoint but shares templates and comes from the historical RL train200 pool.',
            'Old dev52 remains historical loss diagnostic; new dev14 is a separate capability diagnostic.',
            'Whole-dialogue microbatch1 mean assistant CE gives each dialogue one mean loss; token share is not its sampling weight.',
            'Training size and update count change with data; this is a practical repair, not a matched-step data ablation.',
            'Selection60 outcome semantics remain under adjudication; no model improvement established.'])
    (temp/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    temp.rename(args.output)
    print(json.dumps({k:v for k,v in report.items() if k not in ('held_sources','files')},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',type=Path,default=Path('data/sft/staged_v2_candidates_20260925'))
    p.add_argument('--review',type=Path,default=Path('results/analysis/deepseek_rubric_v2_candidates_20260925'))
    p.add_argument('--supplemental',type=Path,default=Path('results/analysis/sft_stage_a_build_20260925/supplemental_candidates.jsonl'))
    p.add_argument('--model',type=Path,default=Path('models/Qwen3.5-4B'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repair-cohort',type=Path)
    p.add_argument('--source-decisions',type=Path)
    args=p.parse_args()
    if args.repair_cohort:
        if not args.source_decisions:
            p.error('--repair-cohort requires --source-decisions')
        finalize_repair(args)
    else:
        finalize(args)


if __name__=='__main__':
    main()
