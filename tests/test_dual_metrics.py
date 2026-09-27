import copy
import json
from types import SimpleNamespace

import pytest

from tau3_grpo.evaluation.dual_metrics import VERSION, summarize_dual, case_record


def row(strict=0, equivalent=1):
    return dict(task_id='T',trial=0,seed=42,reward=equivalent,termination_reason='user_stop',
        simulation={'info':{'outcome_contract':dict(dual_metric_version=VERSION,
        reference_compliant_reward=strict,equivalent_compliant_reward=equivalent,
        outcome_match=True,matched_variants=['alternate'])}})


def test_same_attempt_dual_scores_and_nesting():
    r=row();s=summarize_dual(planned=[dict(task_id='T',trial=0,seed=42)],results=[r],errors=[],trials=1)
    assert s['metrics']['reference_compliant']['solve_rate']==0
    assert s['metrics']['equivalent_compliant']['solve_rate']==1
    assert s['equivalent_only_successes']==1
    with pytest.raises(ValueError,match='nesting'):
        summarize_dual(planned=[dict(task_id='T',trial=0,seed=42)],results=[row(1,0)],errors=[],trials=1)


def test_unknown_keeps_both_denominators_and_no_aggregate():
    plan=[dict(task_id=t,trial=0,seed=42) for t in ['T','U']]
    e=dict(plan[1],error_type='CommunicationUnresolved',error='Simulator scope unknown')
    s=summarize_dual(planned=plan,results=[row()],errors=[e],trials=1)
    for v in s['metrics'].values():
        assert v['planned_trajectories']==2 and v['metrics'] is None and v['solve_rate'] is None
    assert case_record(e,scored=False)['reference_compliant'] is None
    assert case_record(e,scored=False)['labels']==['simulator_scope_unresolved']


def test_no_legacy_renaming_or_duplicate_trial():
    r=row();r['simulation']['info']={}
    with pytest.raises(ValueError,match='rename legacy'):
        summarize_dual(planned=[dict(task_id='T',trial=0,seed=42)],results=[r],errors=[],trials=1)
    with pytest.raises(ValueError,match='duplicate'):
        summarize_dual(planned=[dict(task_id='T',trial=0,seed=42)],results=[row(),row()],errors=[],trials=1)


@pytest.mark.parametrize('state,passed,violations,reason,expected',[
    ('reference',True,[],'user_stop',(1,1)),
    ('alternative',True,[],'user_stop',(0,1)),
    ('unknown',True,[],'user_stop',(0,0)),
    ('reference',False,[],'user_stop',(0,0)),
    ('reference',True,[{'violation':'baggage_removed'}],'user_stop',(0,0)),
    ('reference',True,[],'max_steps',(0,0)),
])
def test_both_metrics_share_all_gates(monkeypatch,tmp_path,state,passed,violations,reason,expected):
    from tau3_grpo.evaluation import outcome_contract as oc
    from tau3_grpo.evaluation.communication_contract import VERSION as CV, requirements
    db=tmp_path/'db.json';db.write_text('{}')
    c=dict(db_hash=oc.sha256_file(db),accepted_outcomes=[{'outcome_sha256':s,'variant':s} for s in ['reference','alternative']],
        strict_outcome_sha256='reference',dual_metric_version=VERSION,communication={'requirements':['Explain completion.']})
    b=dict(version=CV,passed=passed,messages_sha256=oc.sha256_json([]),
        requirements_sha256=oc.sha256_json(requirements([],c)),contract_sha256=oc.sha256_json(c['communication']))
    monkeypatch.setattr(oc,'replay_outcome',lambda *a,**k:dict(outcome_sha256=state,policy_violations=violations))
    sim=SimpleNamespace(model_dump=lambda **k:{'messages':[]},termination_reason=SimpleNamespace(value=reason),
        reward_info=SimpleNamespace(reward=0,reward_breakdown={'DB':0},reward_basis=['DB']))
    result=oc.score_simulation(sim,db_path=db,task=SimpleNamespace(initial_state=None),contract=c,behavior=b)
    assert (result['reference_compliant_reward'],result['equivalent_compliant_reward'])==expected


def test_model_failure_not_inferred_from_outcome_mismatch():
    r=row(0,0);r['simulation']['info']['outcome_contract']['outcome_match']=False
    assert case_record(r,scored=True)['labels']==['outside_frozen_outcome_set_requires_review']


def test_controller_rejects_expansion():
    from tau3_grpo.evaluation.bounded import validate_plan
    p=dict(version='bounded_dual_selection_v1',trials=1,seed=42,task_count=60,
        wall_seconds=7200,gpus=[0,1],arms=[{'name':n} for n in ['base','sft1','sft3']],
        training=False,formal_evaluation=False,source_sha256={'x':'y'},authorization_sha256='z')
    validate_plan(p)
    for key,value in [('trials',4),('wall_seconds',7201),('training',True),('gpus',[0,1,2])]:
        changed=copy.deepcopy(p);changed[key]=value
        with pytest.raises(ValueError):validate_plan(changed)


def test_dual_bundle_rejects_wrong_reference_and_definition(monkeypatch,tmp_path):
    from tau3_grpo.evaluation import outcome_contract as oc
    from tau3_grpo.evaluation.communication_contract import VERSION as CV,SYSTEM
    monkeypatch.setattr(oc,'runtime_identity',lambda:{'frozen':'yes'})
    e=SimpleNamespace(task_id='T',task_hash='task',db_hash='db',task={'id':'T'})
    item=dict(task_hash='task',db_hash='db',record_sha256=oc.sha256_json(e.task),status='approved',
        communication=dict(version=CV,system_sha256=oc.sha256_json(SYSTEM),requirements=['Done']),
        dual_metric_version=VERSION,strict_outcome_sha256='reference',
        accepted_outcomes=[dict(variant='original_reference',outcome_sha256='reference'),
                           dict(variant='alternative',outcome_sha256='alternative')])
    bundle=dict(version=oc.DUAL_VERSION,user_scope_sha256=oc.sha256_json(oc.USER_SCOPE),
        runtime_identity={'frozen':'yes'},dual_rule=oc.DUAL_RULE,behavior_validation={'passed':True},tasks={'T':item})
    p=tmp_path/'bundle.json';p.write_text(json.dumps(bundle));assert oc.load_bundle(p,[e])
    item['strict_outcome_sha256']='alternative';p.write_text(json.dumps(bundle))
    with pytest.raises(ValueError,match='reference'):oc.load_bundle(p,[e])
    item['strict_outcome_sha256']='reference';bundle['dual_rule']={};p.write_text(json.dumps(bundle))
    with pytest.raises(ValueError,match='definition'):oc.load_bundle(p,[e])


def test_deferred_review_only_accepts_complete_communication_evidence(tmp_path):
    from tau3_grpo.evaluation.bounded import BoundedEvaluation
    controller=object.__new__(BoundedEvaluation)
    controller.plan={'defer_communication_review':True}
    path=tmp_path/'errors.jsonl'
    assert not controller.reviewable_errors(tmp_path)
    row={'execution_evidence':{'stage':'communication_scoring','simulation':{'messages':[{'role':'user'}]}}}
    path.write_text(json.dumps(row)+'\n')
    assert controller.reviewable_errors(tmp_path)
    controller.plan={}
    assert not controller.reviewable_errors(tmp_path)
    controller.plan={'defer_communication_review':True}
    path.write_text(json.dumps(row)+'\n'+json.dumps({'error':'endpoint died'})+'\n')
    assert not controller.reviewable_errors(tmp_path)
    path.write_text(json.dumps({'execution_evidence':{'stage':'communication_scoring'}})+'\n')
    assert not controller.reviewable_errors(tmp_path)


def test_repair_queue_requires_all_four_models_and_calibration():
    from tau3_grpo.evaluation.bounded import validate_plan
    p=dict(version='bounded_repair_selection_v2',trials=1,seed=42,task_count=60,
           wall_seconds=7200,gpus=[0,1],arms=[dict(name=x) for x in ('base','sft1','repair_base','repair_sft1')],
           training=False,formal_evaluation=False,source_sha256={'a':'b'},authorization_sha256='x')
    with pytest.raises(ValueError,match='calibration'):validate_plan(p)
    p['simulator_probe']={'receipt':'probe.json','argv':['probe']};validate_plan(p)
    p['arms'].pop()
    with pytest.raises(ValueError,match='model queue'):validate_plan(p)


def test_four_arm_report_includes_continuation_parent_and_unknown(tmp_path, monkeypatch):
    from tau3_grpo.analysis import dual_comparison as dc
    names=['base','sft1','repair_base','repair_sft1']
    bundle=tmp_path/'bundle.json';bundle.write_text('{}')
    (tmp_path/'task_checklist.json').write_text(json.dumps([{'task_id':'T'}]))
    arms=[]
    for name in names:
        p=tmp_path/name;p.mkdir();(p/'run.json').write_text('{}');arms.append({'name':name,'output':str(p)})
    (tmp_path/'plan.json').write_text(json.dumps({'arms':arms,'bundle':str(bundle)}))
    def report(p):
        unknown=p.name=='repair_sft1'
        plan=[dict(task_id='T',trial=0,seed=42)]
        r=summarize_dual(planned=plan,results=[] if unknown else [row()],
                        errors=[dict(plan[0],error='unknown')] if unknown else [],trials=1)
        c=dict(task_id='T',trial=0,seed=42,reference_compliant=None if unknown else 0,equivalent_compliant=None if unknown else 1)
        return r,[c]
    monkeypatch.setattr(dc,'build_report',report)
    monkeypatch.setattr(dc,'compare_evaluations',lambda a,b,**kw:dict(comparable=False,protocol={},differences=None))
    result=dc.report(tmp_path,tmp_path/'comparison')
    assert set(result['runs'])==set(names)
    assert result['status']=='incomplete'
    assert 'repair_sft1_vs_sft1' in result['comparisons_to_parent']
    md=(tmp_path/'comparison/comparison.md').read_text()
    assert 'repair_sft1 严格/等价' in md and 'None/None' in md


def test_diagnostic_probe_override_requires_new_version_and_fixed_temperature():
    from tau3_grpo.evaluation.bounded import validate_plan
    p=dict(version='bounded_repair_selection_v3',trials=1,seed=42,task_count=60,
        wall_seconds=7200,gpus=[0,1],training=False,formal_evaluation=False,
        source_sha256={'a':'b'},authorization_sha256='x',
        simulator_probe={'argv':['probe'],'receipt':'probe.json'},
        probe_failure_policy='record_and_review_each_trajectory',retry_contract='retry.json',retry_contract_sha256='x',
        arms=[dict(name=n,evaluation={'argv':['eval','--user-temperature','0.7','--policy-temperature','0.7']})
              for n in ('base','sft1','repair_base','repair_sft1')])
    validate_plan(p)
    p['version']='bounded_repair_selection_v2'
    with pytest.raises(ValueError,match='v3'):validate_plan(p)
    p['version']='bounded_repair_selection_v3';p['arms'][0]['evaluation']['argv'][2]='0'
    with pytest.raises(ValueError,match='temperature'):validate_plan(p)


def test_retry_requires_prior_user_evidence_not_reward_or_agent_language(tmp_path):
    from tau3_grpo.evaluation.simulator_retry import validate_request
    from tau3_grpo.utils.hashing import sha256_json
    folder=tmp_path/'base/eval';folder.mkdir(parents=True)
    row=dict(task_id='T',trial=0,seed=42,simulation={'reward_info':{'reward':0},
        'messages':[{'role':'assistant','content':'Use card9999?'},{'role':'user','content':'Use card9999.'}]})
    source=folder/'trajectories.jsonl';source.write_text(json.dumps(row)+'\n')
    req=dict(plan_sha256='p',verdict='confirmed_simulator_drift',arm='base',task_id='T',rejected_attempt=0,
        reason_code='contradicted_explicit_payment_constraint',explanation='User contradicts mandatory card.',reviewer='evidence_review',
        source_file='base/eval/trajectories.jsonl',trajectory_sha256=sha256_json(row),scenario_quote='Only use card1234.',
        user_evidence=[{'message_index':1,'quote':'Use card9999.'}])
    kw=dict(root=tmp_path,plan_hash='p',entries={'T':{'task':{'user_scenario':'Only use card1234.'}}},
            completed=set(),contract={'max_attempts_per_cell':4})
    assert validate_request(req,**kw)==('base','T',1)
    req['user_evidence']=[{'message_index':0,'quote':'Use card9999?'}]
    with pytest.raises(ValueError,match='user utterance'):validate_request(req,**kw)
    req['reason_code']='low_reward'
    with pytest.raises(ValueError,match='simulator-specific'):validate_request(req,**kw)
