"""Meaning-preserving pairs and contrasts for the bounded semantic grammar."""
from types import SimpleNamespace as NS
import pytest
from tau3_grpo.algorithms.anchors.semantic import normalize_utterance
from tau3_grpo.algorithms.anchors.evidence import decision_evidence
from tau3_grpo.algorithms.anchors.encoder import AnchorState,encode_anchor,encode_similarity_candidate,resolve_similarity_candidates
from tau3_grpo.envs.registry import SESSIONS,SessionEntry
from tau3_grpo.integrations.anchor_hook import current_anchor

PAIRS=[
 ('user','yes','Yes, please proceed.'),
 ('user','Please go ahead.','I approve.'),
 ('user','No.','Please do not proceed.'),
 ('user','Wait.','Please hold on.'),
 ('user','I revoke my approval.','I withdraw my consent.'),
 ('user','My user ID is alice_123.','User identifier is alice_123.'),
 ('user','My booking code is ABC123.','Reservation ID is ABC123.'),
 ('user','Please cancel my booking ABC123.','I would like to cancel reservation ABC123.'),
 ('user','Can you cancel the booking ABC123 due to a change of plans?','Cancel reservation ABC123 because of change of plans.'),
 ('user','Yes, please go ahead and search for the economy price for that flight so we can see the refund amount.','Yes, check the economy price for the flight.'),
 ('assistant','Hi! How can I help you today?','Hello, how may I assist you today?'),
 ('assistant','Could you provide your user ID?','Please share your user identifier.'),
 ('assistant','Can you share your booking code?','Please provide your reservation ID.'),
 ('assistant','Cancel reservation ABC123 for a $100.00 refund to card card_1. Do you confirm?','Please confirm cancellation of booking ABC123; refund USD 100 to payment method card_1.'),
]


def history(role,text):return [NS(role=role,content=text,tool_calls=None)]


@pytest.mark.parametrize('role,a,b',PAIRS)
def test_equivalent_utterances_merge_without_erasing_scope(role,a,b):
    assert normalize_utterance(role,a)==normalize_utterance(role,b)
    assert normalize_utterance(role,a)['kind']!='opaque'
    assert decision_evidence(history(role,a),version='v3').digest==decision_evidence(history(role,b),version='v3').digest
    assert decision_evidence(history(role,a),version='v2').digest!=decision_evidence(history(role,b),version='v2').digest


@pytest.mark.parametrize('text',[
 'Yes?', 'I approve?', 'My user ID is alice_123?',
 'Yes, but only if there is no fee.', 'Yes, cancel reservation B instead.',
 'Yes, please use a different card.', 'If you verify my name, I will approve.',
 'Please confirm that the bags are free.', 'Do I have to confirm?',
 'Yes, that is my name.', 'No change fees please, then proceed.',
 'I would like to cancel booking ABC123 except the return flight.',
 'Please cancel booking ABC123 and send a certificate.',
 'My user ID is alice_123 but the booking is not mine.',
])
def test_unknown_clauses_are_kept_in_full(text):
    normalized=normalize_utterance('user',text)
    assert normalized['kind']=='opaque' and normalized['text']==text
    assert normalized!=normalize_utterance('user','yes')


@pytest.mark.parametrize('replacement', ['ABC124','USD 200','EUR 100','card_2'])
def test_target_money_currency_payment_changes_do_not_merge(replacement):
    base='Please confirm cancellation of booking ABC123; refund USD 100 to payment method card_1.'
    if replacement=='ABC124':other=base.replace('ABC123',replacement)
    elif replacement=='card_2':other=base.replace('card_1',replacement)
    else:other=base.replace('USD 100',replacement)
    assert normalize_utterance('assistant',base)!=normalize_utterance('assistant',other)


def test_yes_scope_is_bound_in_live_hook_and_protocols_stay_separate():
    def anchor(proposal,reply,version):
        session=NS(task_id='task',db_hash=lambda:'db',messages=history('assistant',proposal)+history('user',reply))
        SESSIONS.register('v3-test',SessionEntry(session=session,anchor_version=version))
        try:return current_anchor(NS(request_id='v3-test'),'assistant')
        finally:SESSIONS.pop('v3-test')
    p='Cancel reservation ABC123 for a $100 refund to card card_1. Do you confirm?'
    assert anchor(p,'yes','v3')==anchor(p,'Yes, please proceed.','v3')
    assert anchor(p,'yes','v3')!=anchor(p.replace('$100','$200'),'yes','v3')
    assert anchor(p,'yes','v3')!=anchor(p,'yes','v2')
    assert anchor(p,'yes','v3').startswith('structured:v3:')


def test_read_and_write_permissions_are_distinct():
    assert normalize_utterance('user','Yes, check the economy price for the flight.')!=normalize_utterance('user','Yes, please proceed.')


def test_id_case_and_missing_slots_are_preserved():
    assert normalize_utterance('user','Please cancel booking ABC123')!=normalize_utterance('user','Please cancel booking abc123')
    assert normalize_utterance('user','Please cancel my booking')!=normalize_utterance('user','Please cancel booking ABC123')
    assert normalize_utterance('assistant','Cancel ABC123 for a refund, yes?')['kind']=='opaque'


def test_v3_similarity_keeps_hard_semantic_guard():
    a=AnchorState('t','db',anchor_version='v3',decision_evidence_hash='a')
    b=AnchorState('t','db',anchor_version='v3',decision_evidence_hash='b')
    ids=resolve_similarity_candidates([encode_similarity_candidate(a),encode_similarity_candidate(b)],threshold=0)
    assert ids[0]!=ids[1]
