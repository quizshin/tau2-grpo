"""Materialize source-first reuse queues with existing evidence, no paid calls.

Does not silently turn old judge labels into intrinsic difficulty or training
approval. Preserves complete conversations and original assistant target mask.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tau3_grpo.analysis.rubric_pilot import visible_events
from tau3_grpo.data.sft import load_complete_airline_dialogues
from tau3_grpo.utils.hashing import sha256_json, sha256_file


def run(args):
    if args.output.exists():raise FileExistsError('Never overwrite a reuse inventory')
    report=json.loads((args.distribution/'distribution.json').read_text())
    if sha256_file(args.source)!=report['source_sha256']:
        raise ValueError('Source changed since distribution census')
    args.output.mkdir(parents=True)
    queue=[json.loads(l) for l in (args.distribution/'source_reuse_queue.jsonl').read_text().splitlines()]
    candidates={r['source_dialog_id']:r for r in queue if not r['blocking_flags']}
    ds,_=load_complete_airline_dialogues(args.source)
    rows=[]; summaries=[]
    for dialogue in ds:
        sid=dialogue.source_dialog_id
        if sid not in candidates:continue
        evidence=candidates[sid];row=dialogue.to_record()
        row['messages']=[{k:v for k,v in m.items() if k in ('role','content','tool_calls','tool_call_id','name')}
                         for m in row['messages']]
        review_path=args.review/'records'/f'{sid}.json'
        status='unreviewed_semantic_candidate';review_hash=None;difficulty=None
        if review_path.exists():
            review=json.loads(review_path.read_text());review_hash=sha256_file(review_path)
            if review['visible_hash']!=sha256_json(visible_events(row['messages'])):
                status='stale_review_requires_recheck'
            elif review['status']!='review_ready':status='prior_review_unresolved'
            else:
                a=review['audit']; difficulty=a.get('difficulty')
                if (a['recommendation']=='keep_candidate' and not a['issues']
                        and all(r['status'] in ('satisfied','not_applicable') for r in a['requirements'])):
                    status='prior_judge_passed_candidate_not_replay_verified'
                else:status='prior_semantic_review_hold'
        f=evidence['features']; tags=[]
        for op in ('flight_or_cabin_change','cancellation','passenger_change'):
            if op in f['operation_tags']:tags.append(op)
        if f['tool_calls']<=6:tags.append('short_observed_trace_not_intrinsic_easy')
        if f['first_tool']=='get_reservation_details':tags.append('reservation_first')
        if 'search_onestop_flight' in f['tool_presence']:tags.append('connection_search')
        if 'calculate' in f['tool_presence']:tags.append('explicit_calculation_tool')
        row['metadata'].update(status=status,evidence_level='source_static_and_existing_review_only',
             source_labels=evidence['source_labels'],previous_review_sha256=review_hash,
             prior_trajectory_difficulty=difficulty,distribution_features=f,
             reuse_priority_tags=tags,already_in_train100=evidence['already_in_train100'],
             exact_task_db_mapping=False,distribution_inventory_sha256=sha256_file(args.distribution/'distribution.json'))
        row['supervision']={'version':'approved_assistant_v1','message_indices':evidence['source_target_indices'],
                            'basis':'original_source_answer_positions_candidate_not_training_approved'}
        rows.append(row)
        summaries.append({'source_dialog_id':sid,'status':status,'already_in_train100':evidence['already_in_train100'],
                          'priority_tags':tags,'source_target_count':len(evidence['source_target_indices'])})
    # Priority review ordering uses only training-side data; no final IDs/answers.
    statuses={'prior_judge_passed_candidate_not_replay_verified':0,'unreviewed_semantic_candidate':1}
    rows.sort(key=lambda r:(statuses.get(r['metadata']['status'],2),
        -len(r['metadata']['reuse_priority_tags']),sha256_json(r['metadata']['source_dialog_id'])))
    for name,selected in [('all_reuse_candidates',rows),('previously_reviewed_candidates',
            [r for r in rows if r['metadata']['status']=='prior_judge_passed_candidate_not_replay_verified'])]:
        (args.output/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    (args.output/'queue.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in summaries))
    report={'count':len(rows),'statuses':dict(Counter(r['status'] for r in summaries)),
            'new_previously_reviewed_candidates':sum(r['status']=='prior_judge_passed_candidate_not_replay_verified' and not r['already_in_train100'] for r in summaries),
            'priority_tags':dict(Counter(t for r in summaries for t in r['priority_tags'])),
            'source_sha256':sha256_file(args.source),'fresh_paid_calls':0,'training_approved':False,
            'final_task_text_or_answers_used':False,'files':{p.name:sha256_file(p) for p in args.output.glob('*.jsonl')}}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data/raw/areal_tau2/tau2_sft_train.jsonl'))
    p.add_argument('--distribution',type=Path,required=True)
    p.add_argument('--review',type=Path,default=Path('results/analysis/deepseek_rubric_v2_candidates_20260925'))
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
