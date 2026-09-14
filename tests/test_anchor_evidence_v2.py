"""Live-hook CPU regression tests for versioned decision evidence."""
from dataclasses import replace
from types import SimpleNamespace as NS
import asyncio
import json

import pytest

from tau3_grpo.algorithms.anchors.evidence import decision_evidence
from tau3_grpo.algorithms.anchors.encoder import (
    AnchorState, AnchorMode, encode_anchor, encode_similarity_candidate,
    resolve_similarity_candidates, encode_anchors,
)
from tau3_grpo.envs.registry import SESSIONS, SessionEntry
from tau3_grpo.integrations.anchor_hook import current_anchor


def msg(role, text='', **kwargs):
    return NS(role=role, content=text, tool_calls=None, **kwargs)


def read(order='A', price=100, call_id='id', error=False):
    return [NS(role='assistant', content=None, tool_calls=[NS(id=call_id,name='get_reservation_details',arguments={'reservation_id':order})]),
            msg('tool',json.dumps({'reservation_id':order,'price':price}),id=call_id,error=error)]


def live(messages, version='v2', mode=AnchorMode.STRUCTURED, request='test-v2', task='task'):
    session=NS(messages=messages,task_id=task,db_hash=lambda:'unchanged-db')
    SESSIONS.register(request,SessionEntry(session=session,anchor_version=version,anchor_mode=mode))
    try:return current_anchor(NS(request_id=request),'assistant')
    finally:SESSIONS.pop(request)


@pytest.mark.parametrize('reply', [
    'I do not confirm this cancellation.', 'Yes, that is my name.',
    'What happens if I confirm?', 'Please wait, do not proceed.',
    'Once you verify the passenger name, I will approve the change.',
    'Yes, please search for the price first.',
])
def test_no_global_consent_from_keywords_and_different_decisions(reply):
    proposal=msg('assistant','Cancel reservation A for a $100 refund to card 1234. Do you confirm?')
    a=[proposal,msg('user',reply)]
    approved=[proposal,msg('user','Yes, please proceed.')]
    assert live(a)!=live(approved)
    assert 'user_confirmed' not in decision_evidence(a).payload()
    assert decision_evidence(a).reply_kind != 'assent_text'


@pytest.mark.parametrize('proposal', [
    'Cancel reservation B for a $100 refund to card 1234. Do you confirm?',
    'Cancel reservation A for a $200 refund to card 1234. Do you confirm?',
    'Cancel reservation A for a €100 refund to card 1234. Do you confirm?',
    'Cancel reservation A for a $100 refund to card 5678. Do you confirm?',
    'Update reservation A to economy with a $100 refund to card 1234. Do you confirm?',
    'Search reservation A for a $100 fare. Shall I search first?',
])
def test_same_yes_is_bound_to_entire_proposal(proposal):
    base=[msg('assistant','Cancel reservation A for a $100 refund to card 1234. Do you confirm?'),msg('user','yes')]
    other=[msg('assistant',proposal),msg('user','yes')]
    assert decision_evidence(base).proposal_hash != decision_evidence(other).proposal_hash
    assert live(base)!=live(other)


def test_withdrawal_and_new_proposal_do_not_reuse_old_exchange():
    base=[msg('assistant','Cancel A?'),msg('user','Yes, please proceed.')]
    revoked=base+[msg('user','Wait, I withdraw consent.')]
    assert decision_evidence(revoked).reply_hash!=decision_evidence(base).reply_hash
    assert live(base)!=live(revoked)
    changed=base+[msg('assistant','Actually, cancel B for $200?')]
    assert decision_evidence(changed).reply_hash is None
    assert live(changed)!=live(base)


def test_earlier_user_constraint_is_not_forgotten_by_later_yes():
    tail=[msg('assistant','Cancel A?'),msg('user','yes')]
    assert live([msg('user','Only if there is no fee.')]+tail)!=live([msg('user','A fee is okay.')]+tail)


def test_read_results_arguments_errors_and_writes_are_state():
    assert live(read('A'))!=live(read('B'))
    assert live(read('A',100))!=live(read('A',200))
    assert live(read(error=True))!=live(read(error=False))
    assert live(read(error=True))!=live([])
    # Sequential different results must not collapse back to an earlier state.
    assert live(read(price=100)+read(price=200)+read(price=100))!=live(read(price=100))
    write=[NS(role='assistant',content=None,tool_calls=[NS(id='w',name='cancel_reservation',arguments={'reservation_id':'A'})]),msg('tool','{"status":"cancelled"}',id='w')]
    assert live(read()+write)!=live(read())


def test_redundant_reads_call_ids_and_json_key_order_do_not_split():
    a=read();b=read(call_id='newid')
    b[0].tool_calls[0].arguments=' {"reservation_id": "A"} '
    b[1].content='{"price":100,"reservation_id":"A"}'
    assert live(a)==live(b)==live(a+b)
    assert live(read('A')+read('B'))==live(read('B')+read('A'))


def test_visible_text_normalization_only_removes_whitespace():
    assert live([msg('user','Cancel  A\nplease')])==live([msg('user','Cancel A please')])
    assert live([msg('user','Cancel A')])!=live([msg('user','Cancel a')])


def test_dict_message_api_shape_and_unmatched_observation():
    structured=[{'role':'assistant','content':None,'tool_calls':[{'id':'call','function':{'name':'get_reservation_details','arguments':'{"reservation_id":"A"}'}}]},
                {'role':'tool','tool_call_id':'call','content':'{"reservation_id":"A","price":100}'}]
    assert decision_evidence(structured).digest==decision_evidence(read()).digest
    assert live([msg('tool','failed to resolve order',id='missing',error=True)])!=live([])


def test_v1_exact_compatibility_and_db_only_ablation():
    from tau3_grpo.algorithms.anchors.features import confirmation_flags,known_info_mask,policy_precondition_flags,last_observation_type
    messages=read()+[msg('user','yes')]
    state=AnchorState('task','unchanged-db',known_info_mask(messages),confirmation_flags(messages),policy_precondition_flags(messages),last_observation_type(messages))
    assert live(messages,'v1')==encode_anchor(state)
    assert live(messages).startswith('structured:v2:')
    assert live(messages)!=live(messages,'v1')
    assert live(messages,mode=AnchorMode.DB_HASH_ONLY)==live([],version='v1',mode=AnchorMode.DB_HASH_ONLY)


def test_similarity_cannot_merge_changed_scope_or_db_even_at_zero_threshold():
    a=AnchorState('task','db',anchor_version='v2',decision_evidence_hash='A')
    b=replace(a,decision_evidence_hash='B');c=replace(a,db_hash='different')
    states=[a,b,c,a]
    ids=resolve_similarity_candidates([encode_similarity_candidate(s) for s in states],threshold=0.)
    assert ids[0]==ids[3] and len(set(ids))==3
    ids,_=encode_anchors(states,mode='similarity',similarity_threshold=0.)
    assert ids[0]==ids[3] and len(set(ids))==3
    mixed=resolve_similarity_candidates([encode_similarity_candidate(a),encode_similarity_candidate(AnchorState('task','db'))],threshold=0.)
    assert mixed[0]!=mixed[1]


def test_live_hook_uses_only_pre_action_session_and_has_no_shared_state():
    from verl.experimental.agent_loop.tool_agent_loop import set_tau3_anchor_hook,tau3_anchor_hook
    session=NS(messages=[msg('user','Cancel A')],task_id='task',db_hash=lambda:'db')
    entry=SessionEntry(session=session,anchor_version='v2');SESSIONS.register('hook-v2',entry)
    data=NS(request_id='hook-v2',anchor_ids=[],anchor_spans=[],response_ids=[999])
    set_tau3_anchor_hook(current_anchor)
    try:
        before=current_anchor(data,'assistant');tau3_anchor_hook(data,'assistant',0,2)
        assert data.anchor_ids==[before] and data.anchor_spans==[(0,2)]
        session.messages.extend(read())
        tau3_anchor_hook(data,'assistant',3,5)
        assert data.anchor_ids[-1]!=before
        tau3_anchor_hook(data,'tool',5,7);assert data.anchor_ids[-1] is None
    finally:set_tau3_anchor_hook(None);SESSIONS.pop('hook-v2')


def test_concurrent_sessions_deterministic_and_isolated():
    async def run():
        async def work(i):
            text=[msg('user',f'Cancel {i}')];expected=live(text,request=f'v2-{i}')
            for _ in range(4):
                await asyncio.sleep(0)
                assert live(text,request=f'v2-{i}')==expected
            return expected
        return await asyncio.gather(*(work(i) for i in range(16)))
    assert len(set(asyncio.run(run())))==16


def test_profile_versions_are_explicit_and_forwarded():
    from tau3_grpo.launch import prepare
    from tau3_grpo.paths import CODE_ROOT
    env={'TAU3_ROOT':'/test','TAU3_RUN_ROOT':'/test/runs'}
    for name,version in [('qwen35_4b_full_a800_c50_matched6h_e2_20260912.yaml','v1'),('qwen35_4b_full_a800_anchor_v2_20260914.yaml','v2')]:
        _,values,_=prepare('rl',CODE_ROOT/'configs/train/rl'/name,'e2',42,[],env)
        assert values['TAU3_GRPO_ANCHOR_VERSION']==version


def real_cases():
    from pathlib import Path
    return json.loads((Path(__file__).parent/'fixtures/anchor_v2_real_cases.json').read_text())


@pytest.mark.parametrize('case',real_cases(),ids=lambda x:x['review_label'])
def test_reviewed_real_replies_no_longer_collapse_into_unconditional_yes(case):
    actual=[msg('assistant',case['previous_assistant']),msg('user',case['user'])]
    unconditional=[actual[0],msg('user','Yes, please proceed.')]
    assert live(actual,'v1')==live(unconditional,'v1')
    assert live(actual,'v2')!=live(unconditional,'v2')


def test_run_protocol_pin_prevents_silent_resume_changes(tmp_path):
    from tau3_grpo.algorithms.anchors.protocol import pin_protocol
    path=pin_protocol(tmp_path,'v2');assert pin_protocol(tmp_path,'v2')==path
    with pytest.raises(ValueError,match='protocol changed'):pin_protocol(tmp_path,'v1')
    with pytest.raises(ValueError,match='unsupported'):pin_protocol(tmp_path,'unknown')


def test_historical_run_protocol_stays_v1(tmp_path):
    from tau3_grpo.algorithms.anchors.protocol import pin_protocol
    (tmp_path/'experiment_manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='historical'):pin_protocol(tmp_path,'v2')
    assert pin_protocol(tmp_path,'v1').exists()
