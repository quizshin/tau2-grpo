"""Read-only source inventory; eligibility is not semantic or training approval."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

BUSINESS = {'book_reservation', 'cancel_reservation', 'update_reservation_flights',
            'update_reservation_baggages', 'update_reservation_passengers',
            'send_certificate', 'transfer_to_human_agents'}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def owner_evidence(text, calls, db):
    """Resolve exact IDs and reservation owners, retaining conflicts and missing IDs."""
    tokens = set(re.findall(r'[A-Za-z0-9_]+', text))
    evidence = []
    missing = []
    for uid in sorted(tokens & db['users'].keys()):
        evidence.append(dict(user_id=uid, basis='literal_user_id'))
    for rid in sorted(tokens & db['reservations'].keys()):
        evidence.append(dict(user_id=db['reservations'][rid]['user_id'],
                             basis='reservation_owner', reservation_id=rid))
    for call in calls:
        args = call['arguments']
        uid = args.get('user_id')
        if uid:
            if uid in db['users']:
                evidence.append(dict(user_id=uid, basis='tool_user_argument'))
            else:
                missing.append(dict(kind='user', value=uid))
        rid = args.get('reservation_id')
        # Newly created reservations can be absent initially: retain as a review flag.
        if rid and rid not in db['reservations']:
            missing.append(dict(kind='reservation_not_in_initial_db', value=rid))
    return evidence, missing


def calls_from_dialogue(row):
    calls = []
    for message in row['messages']:
        if message['role'] != 'assistant':
            continue
        for call in message.get('tool_calls') or []:
            f = call.get('function', call)
            args = f['arguments']
            calls.append(dict(name=f['name'], arguments=json.loads(args) if isinstance(args, str) else args))
    return calls


def replay_candidates(inventory, out):
    """Native CPU tool feasibility only; never claims policy/scenario success."""
    from tau3_grpo.data.manifest import read_manifest
    from tau3_grpo.data.schema import ArealTaskRecord
    from tau3_grpo.envs.adapter import adapt_record
    from tau3_grpo.evaluation.outcome_contract import execute_actions, action_policy_flags, outcome_hash
    summary = json.loads((inventory/'summary.json').read_text())
    for name, expected in summary['source_sha256'].items():
        assert digest(Path(name)) == expected, name
    selected = json.loads((inventory/'reference_replay_queue.json').read_text())
    entries = {e.task_id:e for e in read_manifest('data/manifests/areal_airline_train_seed42.jsonl')}
    results = []
    for tid in selected:
        entry = entries[tid]
        task = adapt_record(ArealTaskRecord.model_validate(entry.task))
        assert digest(Path(task.db_path)) == entry.db_hash
        flags = []
        def observe(env, action):
            flags.extend(dict(action=action, violation=f) for f in action_policy_flags(env, action))
        row = dict(task_id=tid, db_hash=entry.db_hash, task_hash=entry.task_hash, semantic_review='pending', training_ready=False)
        try:
            env, receipts = execute_actions(task.db_path, entry.task['evaluation_criteria']['actions'],
                                            task.task.initial_state, before_action=observe)
            row.update(tool_execution='passed', receipts=receipts, outcome_sha256=outcome_hash(env))
        except Exception as exc:
            row.update(tool_execution='failed', error=type(exc).__name__+': '+str(exc))
        row['limited_policy_flags'] = flags
        results.append(row)
    (out/'cases.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in results))
    result = dict(tasks=len(results), tool_execution_counts=dict(Counter(x['tool_execution'] for x in results)),
        cases_with_limited_policy_flags=sum(bool(x['limited_policy_flags']) for x in results),
        new_api_calls=0, gpu_used=False, training_ready_count=0,
        inventory_summary_sha256=digest(inventory/'summary.json'),
        limitation='Native tool execution is feasibility only. Policy checks cover three structural cases, not eligibility, semantic goals, finance, consent or source dialogue faithfulness.')
    write(out/'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay-inventory', type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    if args.replay_inventory:
        replay_candidates(args.replay_inventory, out)
        return
    plan_path = Path('data/sft/decision_repair_20260926_v1/plan.json')
    plan = json.loads(plan_path.read_text())
    protected = set(plan['protected_users']) | set(plan['validation_users'])
    train_path = Path('data/manifests/areal_airline_train_seed42.jsonl')
    source_path = Path('data/sft/source_reuse_20260926_v2/all_reuse_candidates.jsonl')
    entries = rows(train_path)
    dbs = {}
    hashes = {}
    for entry in entries:
        p = Path('data/raw/areal_tau2') / entry['db_path']
        if str(p) not in dbs:
            dbs[str(p)] = json.loads(p.read_text())
            hashes[str(p)] = digest(p)
        assert hashes[str(p)] == entry['db_hash'], entry['task_id']
    task_rows = []
    for line, entry in enumerate(entries, 1):
        assert entry['split'] == 'train'
        task = entry['task']
        calls = task['evaluation_criteria']['actions']
        names = [x['name'] for x in calls if x['name'] in BUSINESS]
        db = dbs[str(Path('data/raw/areal_tau2') / entry['db_path'])]
        evidence, missing = owner_evidence(json.dumps(task), calls, db)
        owners = {x['user_id'] for x in evidence}
        overlap = owners & protected
        reasons = []
        if not owners: reasons.append('unresolved_user')
        if len(owners) > 1: reasons.append('multiple_user_identities_require_review')
        if overlap: reasons.append('protected_user_overlap')
        if missing: reasons.append('unresolved_referenced_entity')
        if task.get('initial_state'): reasons.append('initial_state_requires_review')
        task_rows.append(dict(task_id=entry['task_id'], source_line=line, task_hash=entry['task_hash'],
            db_path=entry['db_path'], db_hash=entry['db_hash'], owners=sorted(owners),
            owner_evidence=evidence, unresolved_entities=missing, protected_overlap=sorted(overlap),
            business_action_count=len(names), business_actions=names,
            distinct_action_types=len(set(names)), pattern=' + '.join(names),
            identity_gate='blocked' if reasons else 'passed', gate_reasons=reasons,
            status='identity_eligible_semantic_pending' if not reasons else 'held',
            training_ready=False))
    source_rows = []
    for line, row in enumerate(rows(source_path), 1):
        calls = calls_from_dialogue(row)
        names = [x['name'] for x in calls if x['name'] in BUSINESS]
        # Union across possible DBs is conservative; it does not assert exact DB linkage.
        text = json.dumps([m for m in row['messages'] if m['role'] != 'system'])
        owners = set()
        for db in dbs.values():
            evidence, _ = owner_evidence(text, calls, db)
            owners.update(x['user_id'] for x in evidence)
        metadata = row['metadata']
        source_rows.append(dict(source_dialog_id=metadata['source_dialog_id'], source_line=line,
            dialogue_hash=metadata['dialogue_hash'], owners=sorted(owners),
            protected_overlap=sorted(owners & protected), owner_resolution='possible_db_union',
            business_action_count=len(names), distinct_action_types=len(set(names)), pattern=' + '.join(names),
            previous_review_status=metadata['status'], already_in_train100=metadata['already_in_train100'],
            exact_task_db_mapping=metadata.get('exact_task_db_mapping', False),
            candidate_identity_gate='passed_conservative_screen' if len(owners)==1 and not owners & protected else 'held',
            status='requires_exact_source_mapping_and_semantic_review', training_ready=False))
    assert len({x['task_id'] for x in task_rows}) == len(task_rows) == 200
    assert len({x['source_dialog_id'] for x in source_rows}) == len(source_rows)
    for name, values in [('train_task_inventory', task_rows), ('source_dialogue_inventory', source_rows)]:
        (out / (name+'.jsonl')).write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in values))
    candidates = [x for x in task_rows if x['business_action_count']>1 and x['identity_gate']=='passed']
    sources = [x for x in source_rows if x['business_action_count']>1 and x['candidate_identity_gate']=='passed_conservative_screen']
    summary = dict(version='multistep_source_inventory_v1', train_tasks=len(task_rows),
        train_multiaction=sum(x['business_action_count']>1 for x in task_rows),
        train_multiaction_identity_eligible=len(candidates),
        train_multiaction_block_reasons=dict(Counter(r for x in task_rows if x['business_action_count']>1 for r in x['gate_reasons'])),
        eligible_patterns=dict(Counter(x['pattern'] for x in candidates)),
        eligible_users=len({u for x in candidates for u in x['owners']}),
        source_dialogues=len(source_rows), source_multiaction=sum(x['business_action_count']>1 for x in source_rows),
        source_multiaction_identity_screened=len(sources),
        source_multiaction_new_to_train100=sum(not x['already_in_train100'] for x in sources),
        source_identity_screened_prior_statuses=dict(Counter(x['previous_review_status'] for x in sources)),
        source_multiaction_screen_reasons={
            'no_resolved_owner':sum(not x['owners'] for x in source_rows if x['business_action_count']>1),
            'multiple_possible_owners':sum(len(x['owners'])>1 for x in source_rows if x['business_action_count']>1),
            'protected_overlap':sum(bool(x['protected_overlap']) for x in source_rows if x['business_action_count']>1)},
        protected_user_count=len(protected), new_api_calls=0, gpu_used=False, training_ready_count=0,
        final_or_reserve_read=False,
        source_sha256={str(plan_path):digest(plan_path),str(train_path):digest(train_path),str(source_path):digest(source_path),**hashes},
        limitations=['Repeated business calls count as actions; this is not calibrated difficulty.',
          'Source dialogue DB identity and tool observations remain unverified. Identity screening is not replay or semantic acceptance.',
          'Protected set reuses frozen selection and held-out user metadata; no final/reserve task content read.',
          'Train candidates require reference replay, scenario-policy consistency, and new family-level validation split.'])
    write(out/'summary.json', summary)
    write(out/'reference_replay_queue.json', [x['task_id'] for x in candidates])
    write(out/'source_review_queue.json', [x['source_dialog_id'] for x in sorted(sources, key=lambda x:(x['already_in_train100'], x['previous_review_status']!='prior_judge_passed_candidate_not_replay_verified',x['source_dialog_id']))])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
