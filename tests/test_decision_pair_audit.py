"""Integrity and outcome-independent execution of the CPU diagnostic harness."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from env_info.a800_20260912.audit_decision_pairs import anchor, audit, validate
from env_info.a800_20260912.build_decision_pair_fixture import state, msg
from tau3_grpo.envs.registry import SESSIONS

FIXTURE=Path(__file__).parent/'fixtures/decision_state_pairs_20260914.json'


def test_reviewed_fixture_has_separate_provenance_and_no_future_fields():
    fixture=json.loads(FIXTURE.read_text());validate(fixture)
    assert fixture['annotation']['independent_human_labels'] is False
    assert fixture['annotation']['outcomes_used_for_labels'] is False
    for p in fixture['pairs']:
        if p['kind']=='natural':
            assert len(p['source_refs'])==2
            assert p['source_refs'][0]['trial']!=p['source_refs'][1]['trial']
            for s in (p['a'],p['b']):
                assert len(s['messages'])==2
                assert not any(m.get('tool_calls') for m in s['messages'])
        if p['kind']=='real_prefix_counterfactual':
            assert p['source_refs'][1]['observed_trajectory'] is False
        if p['kind']=='external_budget':
            assert p['relation']=='uncertain'  # Deliberate abstraction, not a GiGPO violation.


def test_changing_review_labels_cannot_change_anchor_predictions():
    fixture=json.loads(FIXTURE.read_text());fixture['pairs']=fixture['pairs'][:2]
    a=audit(fixture)
    changed=deepcopy(fixture)
    for pair in changed['pairs']:
        pair['relation']='separate';pair['rationale']='new review label'
    b=audit(changed)
    assert [p['prediction'] for p in a['pairs']]==[p['prediction'] for p in b['pairs']]
    assert [p['anchors'] for p in a['pairs']]==[p['anchors'] for p in b['pairs']]


def test_live_hook_preserves_unrelated_session_and_cleans_failed_audit():
    sentinel=object();SESSIONS.register('unrelated-pair-test',sentinel)
    before=SESSIONS.active_count()
    try:
        s=state([msg('user','Hello')])
        assert anchor(s,'v3').startswith('structured:v3:')
        with pytest.raises(ValueError):anchor(s,'invalid')
        assert SESSIONS.active_count()==before
        assert SESSIONS.get('unrelated-pair-test') is sentinel
    finally:SESSIONS.pop('unrelated-pair-test')


@pytest.mark.parametrize('field',['reward','future_action','final_db'])
def test_unexpected_state_fields_rejected(field):
    fixture=json.loads(FIXTURE.read_text());fixture['pairs']=fixture['pairs'][:1]
    fixture['pairs'][0]['a'][field]='not allowed'
    with pytest.raises(ValueError,match='unexpected state field'):validate(fixture)


def test_duplicate_pairs_and_unknown_relation_rejected():
    fixture=json.loads(FIXTURE.read_text());fixture['pairs']=fixture['pairs'][:1]*2
    with pytest.raises(ValueError,match='duplicate'):validate(fixture)
    fixture['pairs']=deepcopy(fixture['pairs'][:1]);fixture['pairs'][0]['relation']='looks close'
    with pytest.raises(ValueError,match='invalid relation'):validate(fixture)


def test_post_label_outcome_inspection_rejects_changed_annotations(tmp_path):
    import hashlib
    from env_info.a800_20260912.audit_decision_pairs import digest
    from env_info.a800_20260912.decision_pair_outcomes import inspect
    fixture=json.loads(FIXTURE.read_text());fixture['pairs']=fixture['pairs'][:1]
    source=tmp_path/'trajectories.jsonl'
    refs=fixture['pairs'][0]['source_refs']
    source.write_text('\n'.join(json.dumps({'task_id':r['task_id'],'trial':r['trial'],'reward':i}) for i,r in enumerate(refs)))
    fixture['sources']['e2/trajectories.jsonl']=hashlib.sha256(source.read_bytes()).hexdigest()
    path=tmp_path/'fixture.json';path.write_text(json.dumps(fixture))
    audited=tmp_path/'audit.json';audited.write_text(json.dumps({'fixture_sha256_canonical':digest(fixture)}))
    assert inspect(path,audited,source)['discordant_pairs']==1
    fixture['pairs'][0]['relation']='separate';path.write_text(json.dumps(fixture))
    with pytest.raises(ValueError,match='Labels changed'):inspect(path,audited,source)
