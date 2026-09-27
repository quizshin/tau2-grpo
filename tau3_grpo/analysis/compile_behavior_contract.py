"""Compile the full repaired selection60: executable outcomes and frozen predicates."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from tau3_grpo.analysis.prepare_outcome_contract import variants,action_policy_flags
from tau3_grpo.data.manifest import read_manifest
from tau3_grpo.data.schema import ArealTaskRecord
from tau3_grpo.envs.adapter import adapt_record
from tau3_grpo.evaluation import outcome_contract as oc
from tau3_grpo.evaluation.communication_contract import VERSION as BEHAVIOR_VERSION,SYSTEM
from tau3_grpo.utils.hashing import sha256_file,sha256_json

EXTRA={
 33:['Explain that Chen Johansson already has the requested surname; do not claim an edit was made.'],
 64:['Explain why work schedule changes are not covered by insurance, then carry out the explicitly requested human review.'],
 852:['Explain cancellation ineligibility, then carry out the explicitly requested human transfer without claiming cancellation.'],
 802:['Explain that keeping the same three passenger records does not split a reservation; decline the requested split.',
      'Explain the one-certificate/three-gift-card per-booking limit and refuse the new booking when compliant alternatives are rejected.'],
 657:['Refuse reducing the passenger counts on the overlapping May19 bookings.',
      'Explain why added bags cannot be removed and why a travel certificate cannot replace the existing baggage payment.',
      'Explain which weekend bookings qualify under the24-hour cancellation rule and leave older ones active.'],
 458:['Explain why the May20 morning BOS-MCO bookings cannot be cancelled under the applicable conditions; do not claim to cancel them.'],
 895:['Explain why older weekend bookings remain active and why the basic-economy May20 outbound cannot be changed; do not claim to change it.'],
 921:['Explain the refusal to remove children from the May22 bookings and do not claim their passenger count was reduced.'],
 902:['Explain why single-passenger removal is prohibited and get consent for the alternative whole-party downgrade.'],
 994:['Explain why single-passenger removal is prohibited and get consent for the alternative whole-party downgrade.'],
 162:['Clarify which one of the three duplicate ORD-ATL reservations will be kept before cancelling the other two.'],
 762:['Confirm keeping the09:30 MSP-DTW booking and cancelling only the two other duplicate bookings.'],
 923:['Clarify the correct round-trip reservation and the chosen later return date; preserve the outbound.'],
}


def communication(record,review):
    tid=int(record.id.split('_')[-1]);req=[
      'Clearly communicate the verified outcome of this task and its required non-database subgoals, including any refusal/no-op, without falsely claiming unperformed actions.',
      'Ground material factual claims, fare/refund explanations and policy decisions in the visible tool results and supplied policy. Do not invent cancellation reasons, passenger identities, or disruption causes.']
    req+=review.get('goal_checks',[])+EXTRA.get(tid,[])
    req=list(dict.fromkeys(req))
    order=[]
    if tid in (490,976):order=[dict(before='update_reservation_flights',before_arguments={'reservation_id':'GCZ58I'},after='send_certificate')]
    if tid==402:order=[dict(before='cancel_reservation',before_arguments={'reservation_id':'SCVMEC'},after='send_certificate')]
    return dict(version=BEHAVIOR_VERSION,system_sha256=sha256_json(SYSTEM),requirements=req,
                required_successful_tools=['transfer_to_human_agents'] if tid in (64,852,1073) else [],
                action_order=order)


def run(args):
    if args.output.exists():raise FileExistsError('Use a new contract directory')
    source=args.manifest_dir/'areal_airline_selection_seed42.jsonl'
    entries=read_manifest(source);reviews={x['task_id']:x for x in map(json.loads,(args.manifest_dir/'changes.jsonl').read_text().splitlines())}
    if len(entries)!=60 or len({e.task_id for e in entries})!=60 or any(e.split!='selection' for e in entries):
        raise ValueError('Expected the complete exposed selection60')
    args.output.mkdir(parents=True)
    dual = getattr(args, 'dual_metrics', False)
    bundle=dict(version=oc.DUAL_VERSION if dual else oc.BEHAVIOR_VERSION,user_scope_sha256=sha256_json(oc.USER_SCOPE),tasks={},
        runtime_identity=oc.runtime_identity(),selection_manifest_sha256=sha256_file(source),
        compiler_sha256=sha256_file(__file__),behavior_model='deepseek-flash',
        budget_directory='results/analysis/deepseek_rubric_pilot_20260925',
        purpose='candidate_full_task_predicates_pending_calibration',final_or_reserve_read=False)
    if dual:
        bundle['dual_rule'] = oc.DUAL_RULE
    previous=json.loads(args.reuse_executions.read_text()) if args.reuse_executions else None
    if previous:
        # Reuse only deterministic native execution receipts, NOT old approvals or
        # communication verdicts. Scorer/communication revisions are recompiled.
        for key in ('prompt','native_sources','adapter_sha256'):
            if previous['runtime_identity'][key]!=bundle['runtime_identity'][key]:
                raise ValueError('Cannot reuse changed native execution identity: '+key)
        if previous['selection_manifest_sha256']!=bundle['selection_manifest_sha256']:
            raise ValueError('Cannot reuse executions for changed questions')
        bundle['reused_executions']=dict(source_sha256=sha256_file(args.reuse_executions),
            scope='native receipts only; exact DB/task/action identity; no old status/communication reused')
    for e in entries:
        r=ArealTaskRecord.model_validate(e.task);adapted=adapt_record(r)
        if r.fingerprint!=e.task_hash or sha256_file(adapted.db_path)!=e.db_hash:raise ValueError('Source drift')
        db=json.loads(adapted.db_path.read_text());item=dict(task_hash=e.task_hash,record_sha256=sha256_json(e.task),
            db_hash=e.db_hash,status='behavior_validation_pending',accepted_outcomes=[],issues=[],
            communication=communication(r,reviews[e.task_id]))
        prior=previous['tasks'][e.task_id] if previous else None
        if prior and (prior['db_hash']!=e.db_hash or prior['task_hash']!=e.task_hash):
            raise ValueError('Stale reusable native execution')
        reusable={x['actions_sha256']:x for x in prior['accepted_outcomes']} if prior else {}
        for label,actions in variants(r,db):
            ah=sha256_json(actions)
            if ah in reusable:
                old=reusable[ah]
                if old['actions']!=actions or sha256_json(old['actions'])!=ah:raise ValueError('Corrupt native receipt')
                item['accepted_outcomes'].append(dict(old,variant=label))
                continue

            flags=[]
            def inspect(env,a):flags.extend(action_policy_flags(env,a))
            try:
                env,receipts=oc.execute_actions(adapted.db_path,actions,adapted.task.initial_state,before_action=inspect)
                if flags:raise ValueError(str(flags))
                item['accepted_outcomes'].append(dict(variant=label,outcome_sha256=oc.outcome_hash(env),actions=actions,
                    actions_sha256=sha256_json(actions),receipts=receipts))
            except Exception as exc:
                # Never silently use an invalid alternative or drop a task.
                item['issues'].append(dict(variant=label,error=str(exc)))
        if dual:
            reference = [x for x in item['accepted_outcomes'] if x['variant'] == 'original_reference']
            if len(reference) != 1:
                raise ValueError('Exactly one repaired reference must be frozen for ' + e.task_id)
            item['strict_outcome_sha256'] = reference[0]['outcome_sha256']
            item['dual_metric_version'] = oc.DUAL_RULE['version']
        if item['issues'] or not item['accepted_outcomes']:item['status']='blocked_quality_issue'
        bundle['tasks'][e.task_id]=item
        print(json.dumps({'task':e.task_id,'done':len(bundle['tasks']),'variants':len(item['accepted_outcomes']),'issues':item['issues']}),flush=True)
    if args.calibration:
        calibration=json.loads(args.calibration.read_text())
        if (calibration.get('passed') is not True or calibration.get('correct')!=13
                or calibration.get('cases')!=13 or calibration.get('system_sha256')!=sha256_json(SYSTEM)):
            raise ValueError('Behavior calibration failed or changed')
        if any(x['issues'] or not x['accepted_outcomes'] for x in bundle['tasks'].values()):
            raise ValueError('Unresolved deterministic task outcomes')
        bundle['behavior_validation']={'passed':True,'calibration_sha256':sha256_file(args.calibration),
            'scope':'13 authored controls; provisional pipeline smoke only, not real-model judge accuracy'}
        bundle['approval_scope']='smoke_only'
        bundle['formal_evaluation_ready']=False
        for item in bundle['tasks'].values():item['status']='approved'
    (args.output/'bundle.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2)+'\n')
    summary=dict(tasks=len(entries),statuses=dict(Counter(x['status'] for x in bundle['tasks'].values())),
        unique_outcomes={k:len({o['outcome_sha256'] for o in v['accepted_outcomes']}) for k,v in bundle['tasks'].items()},
        live_ready=False,smoke_ready=bool(args.calibration),approval_scope=bundle.get('approval_scope'),
        paid_calls=0,gpu_hours=0)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if dual:
        checklist = []
        for task_id,item in bundle['tasks'].items():
            reference=next(o for o in item['accepted_outcomes'] if o['variant']=='original_reference')
            checklist.append(dict(task_id=task_id,task_hash=item['task_hash'],db_hash=item['db_hash'],
                strict_outcome_sha256=item['strict_outcome_sha256'],
                allowed_outcomes=[dict(variant=o['variant'],outcome_sha256=o['outcome_sha256'],
                                      actions_sha256=o['actions_sha256'],actions=o['actions'])
                                  for o in item['accepted_outcomes']],
                reference_actions=reference['actions'],communication=item['communication'],
                base_result=None,sft1_result=None,sft3_result=None,
                result_status='not_run_rules_frozen_before_model_scores'))
        (args.output/'task_checklist.json').write_text(json.dumps(checklist,ensure_ascii=False,indent=2)+'\n')
        lines=['# selection60 双指标逐题规则（运行前）','',
               '严格口径为修订题目的单一参考终态；等价口径为预先执行验证的合法终态集合。',
               '两者使用相同政策、授权、沟通与终止要求；不进行工具调用轨迹逐步匹配。',
               '不能忽略支付历史、航程顺序、身份、日期或未获授权的副作用。',
               '未枚举的疑似合法结果仅进入人工复核，不根据模型得分自动扩充集合。',
               '模拟器漂移、judge不确定及API错误均未评分；不删分母、不重采样掩盖错误。','']
        for x in checklist:
            lines += [f"## {x['task_id']}", '',
                      f"- 参考终态：`{x['strict_outcome_sha256']}`。",
                      f"- 合法终态数：{len({o['outcome_sha256'] for o in x['allowed_outcomes']})}。",
                      '- 允许路径：'+', '.join(o['variant'] for o in x['allowed_outcomes']),
                      '- 必须成功的工具：'+', '.join(x['communication']['required_successful_tools']),
                      '- 业务顺序依赖：'+json.dumps(x['communication']['action_order'],ensure_ascii=False),
                      '- 每次预订写操作均检查事前知情确认；允许一次确认覆盖已充分披露的多项操作。',
                      '- 必要沟通：', *['  - '+s for s in x['communication']['requirements']],
                      '- Base／SFT1／SFT3：待运行。', '']
        (args.output/'task_checklist.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reuse-executions',type=Path)
    p.add_argument('--dual-metrics',action='store_true',help='Freeze nested reference/equivalent compliance metrics before model results')
    p.add_argument('--calibration',type=Path,help='Passing authored controls enable one-trial smoke only, never formal evaluation')
    p.add_argument('--manifest-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
