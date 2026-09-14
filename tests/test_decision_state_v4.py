"""Decision reducer contracts: binding, abstention, and discount preservation."""
from copy import deepcopy
from types import SimpleNamespace as NS
import json

import numpy as np
import pytest

from tau3_grpo.algorithms.anchors.decision_state import ABSTAIN_ANCHOR, assess_messages, parse_event
from tau3_grpo.algorithms.tau_gigpo import steps_from_anchor_payload, step_returns, compute_tau_gigpo_advantage
from tau3_grpo.envs.registry import SESSIONS,SessionEntry
from tau3_grpo.integrations.anchor_hook import current_anchor

PROPOSAL='Change reservation ABC123 from business to economy on flight HAT216 on 2024-05-21. Refund USD 654 to payment method card_1. Do you confirm?'


def m(role,content):return {'role':role,'content':content}


def base():
    calls=[{'id':'u','name':'get_user_details','arguments':{'user_id':'alice'}},
           {'id':'r','name':'get_reservation_details','arguments':{'reservation_id':'ABC123'}}]
    return [m('assistant','Hi! How can I help you today?'),m('user','My user ID is alice.'),
            m('user','Please change reservation ABC123 from business to economy.'),
            {'role':'assistant','content':None,'tool_calls':calls},
            {'role':'tool','id':'u','error':False,'content':json.dumps({'user_id':'alice','payment_methods':{'card_1':{},'card_2':{}}})},
            {'role':'tool','id':'r','error':False,'content':json.dumps({'reservation_id':'ABC123','user_id':'alice','cabin':'business','flights':[{'flight_number':'HAT216','date':'2024-05-21'}]})}]


def ready(reply='Yes, please proceed.'):return base()+[m('assistant',PROPOSAL),m('user',reply)]


def live(messages,request='v4-test',task='t',db='db'):
    from tau3_grpo.analysis.replay_decisions import as_object
    session=NS(messages=[as_object(x) for x in messages],task_id=task,db_hash=lambda:db,policy=lambda:'policy')
    entry=SessionEntry(session=session,anchor_version='v4');SESSIONS.register(request,entry)
    try:return current_anchor(NS(request_id=request),'assistant'),entry.decision_diagnostics
    finally:SESSIONS.pop(request)


def test_equivalent_complete_states_merge_at_live_hook():
    a=ready();b=ready('I approve.')
    b[2]['content']='I would like to downgrade booking ABC123 from business to economy.'
    aa=assess_messages(a);bb=assess_messages(b)
    assert aa.comparable and bb.comparable and aa.digest==bb.digest
    assert live(a)[0]==live(b)[0]
    assert live(a)[0].startswith('structured:v4:')


@pytest.mark.parametrize('old,new',[('USD 654','USD 272'),('USD','EUR'),('card_1','card_2')])
def test_quoted_terms_are_not_replaced_by_actual_db_facts(old,new):
    a=ready();b=deepcopy(a);b[-2]['content']=PROPOSAL.replace(old,new)
    assert assess_messages(a).comparable and assess_messages(b).comparable
    assert live(a)[0]!=live(b)[0]


@pytest.mark.parametrize('reply',['No, do not proceed.','Wait.','I revoke my approval.','Yes, but only if there is no fee.'])
def test_consent_reject_pause_revoke_and_condition_are_distinct(reply):
    a=ready();b=ready()+[m('user',reply)]
    assert assess_messages(b).comparable
    assert live(a)[0]!=live(b)[0]
    assert assess_messages(b).state['decision']['consent']!='approved'


def test_changed_proposal_invalidates_old_approval_and_retains_prior_promise():
    a=ready();changed=PROPOSAL.replace('654','272')
    b=a+[m('assistant',changed)]
    state=assess_messages(b)
    assert state.comparable and state.state['decision']['approved_proposal'] is None
    assert state.state['decision']['consent']=='pending'
    assert state.state['decision']['history_guard'][-1]['prior_proposal']['quoted_refund']=='654'
    direct=base()+[m('assistant',changed),m('user','Yes')]
    assert live(b+[m('user','Yes')])[0]!=live(direct)[0]


def test_intervening_question_or_payment_change_cannot_reuse_yes():
    for extra in [m('assistant','Could you provide your user ID?'),m('user','Please use card card_2 for the refund.')]:
        a=ready()+[extra,m('user','Yes')]
        assert live(a)[0]==ABSTAIN_ANCHOR
        assert assess_messages(a).state['decision']['approved_proposal'] is None


@pytest.mark.parametrize('text',[
 'Yes?', 'Only proceed if the passenger name matches mine.',
 'Please change reservation ABC123 from business to economy and add two bags.',
 'Yes, but use a different card.', 'This booking does not belong to me.',
])
def test_unknown_qualifications_cannot_be_erased_by_later_complete_proposal(text):
    a=base()+[m('user',text),m('assistant',PROPOSAL),m('user','Yes')]
    anchor,diagnostic=live(a)
    assert anchor==ABSTAIN_ANCHOR and 'unparsed_or_unresolved_history' in diagnostic['reasons']


def test_duplicate_goal_does_not_erase_old_financial_promise():
    a=ready()+[m('user','Please change reservation ABC123 from business to economy.'),
               m('assistant',PROPOSAL.replace('654','272')),m('user','Yes')]
    assert assess_messages(a).state['decision']['history_guard'][-1]['prior_proposal']['quoted_refund']=='654'


def test_wrong_identity_flight_scope_and_missing_policy_abstain():
    a=ready();a[5]['content']=a[5]['content'].replace('alice','bob')
    assert live(a)[0]==ABSTAIN_ANCHOR
    b=ready();b[-2]['content']=PROPOSAL.replace('2024-05-21','2024-05-22')
    assert live(b)[0]==ABSTAIN_ANCHOR
    assert live(ready(),db=None)[0]==ABSTAIN_ANCHOR
    assert live(ready(),task='other')[0]!=live(ready())[0]


def test_time_steps_keep_abstentions_but_skip_observation_segments():
    ids=[['structured:v4:a',None,ABSTAIN_ANCHOR,'structured:v4:b']]
    spans=[[(0,1),(1,2),(2,3),(3,4)]]
    steps=steps_from_anchor_payload(ids,spans)
    assert len(steps)==3 and steps[1].anchor_id is None
    np.testing.assert_allclose(step_returns(steps,[1.],gamma=.95),[.95**2,.95,1])
    # No accidental grouping of identical abstention markers; episode remains.
    steps=steps_from_anchor_payload([[ABSTAIN_ANCHOR],[ABSTAIN_ANCHOR]],[[(0,1)],[(0,1)]])
    advantage,stats=compute_tau_gigpo_advantage([1,0],['g','g'],steps,response_length=1)
    np.testing.assert_allclose(advantage,[[.5],[-.5]])
    assert stats.usable_anchor_groups==0


def test_v4_cannot_mix_independent_episode_groups():
    import torch
    from tau3_grpo.algorithms.verl_estimator import compute_tau_gigpo_verl
    r=torch.tensor([[1.],[0.]]);mask=torch.ones_like(r)
    payload={'anchor_ids':[['structured:v4:same'],['structured:v4:same']],'anchor_spans':[[(0,1)],[(0,1)]]}
    adv,_=compute_tau_gigpo_verl(r,mask,index=['group1','group2'],non_tensor_batch=payload)
    assert torch.equal(adv,torch.zeros_like(adv))
    adv,_=compute_tau_gigpo_verl(r,mask,index=['same','same'],non_tensor_batch=payload)
    assert torch.equal(adv,torch.tensor([[1.],[-1.]]))
