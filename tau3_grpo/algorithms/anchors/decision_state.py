"""Bounded decision-state reducer; unsupported language causes abstention.

Only pre-action, visible messages enter this reducer. It does not authorize tool
execution and it never replaces the policy's conversation. Full text coverage
is required: no slot-only regex may erase an unparsed qualification.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal
import re
from typing import Any

from tau3_grpo.algorithms.anchors.evidence import _get, decision_evidence, json_value, text_value
from tau3_grpo.algorithms.anchors.semantic import normalize_utterance
from tau3_grpo.utils.hashing import sha256_json

ABSTAIN_ANCHOR = 'abstain:v4'
ID = r'[A-Za-z0-9_-]+'
CABIN = r'basic economy|economy|business'


def cabin(value):return value.lower().replace(' ', '_')


def event(kind, **data):return {'kind':kind,'data':data}


def parse_event(role: str, text: str) -> dict:
    """A small whole-utterance grammar, deliberately not a general extractor."""
    text=text_value(text).replace('’', "'")
    s=text.rstrip('.!').strip()
    def match(pattern):return re.fullmatch(pattern,s,re.I)
    if role=='user':
        m=match(rf'(?:please |i would like to |i\'d like to )?(?:change|downgrade|upgrade) (?:reservation|booking) (?P<reservation>{ID}) from (?P<source>{CABIN}) to (?P<target>{CABIN})')
        if m:
            d=m.groupdict();return event('goal',operation='change_cabin',reservation_id=d['reservation'],source_cabin=cabin(d['source']),target_cabin=cabin(d['target']))
        m=match(rf'(?:please )?(?:use|refund to) (?:payment method|card) (?P<payment>{ID})(?: for the refund)?')
        if m:return event('payment_constraint',payment_id=m['payment'])
        if match(r'(?:only proceed|proceed only) if there is no fee'):
            return event('condition',condition='no_fee')
        if match(r'yes,? but only if there is no fee'):
            return event('conditional_consent',condition='no_fee')
    if role=='assistant':
        m=match(rf'change (?:reservation|booking) (?P<reservation>{ID}) from (?P<source>{CABIN}) to (?P<target>{CABIN}) on flight (?P<flight>{ID}) on (?P<date>\d{{4}}-\d{{2}}-\d{{2}})\. refund (?P<currency>USD|EUR|GBP) (?P<amount>\d+(?:\.\d+)?) to (?:payment method|card) (?P<payment>{ID})\. (?:do you confirm\?|please confirm)')
        if m:
            d=m.groupdict()
            return event('proposal',operation='change_cabin',reservation_id=d['reservation'],
                         source_cabin=cabin(d['source']),target_cabin=cabin(d['target']),
                         flights=[{'flight_number':d['flight'],'date':d['date']}],
                         quoted_refund=format(Decimal(d['amount']).normalize(),'f'),
                         currency=d['currency'].upper(),payment_id=d['payment'])
    normalized=normalize_utterance(role,text)
    kind=normalized['kind'];slots=normalized.get('slots',{})
    if role=='user':
        if kind=='identity':return event('identity_claim',**slots)
        if kind=='reservation_reference':return event('reservation_reference',**slots)
        if kind=='request_cancellation':return event('goal',operation='cancel',reservation_id=slots['reservation'],reason=slots['reason'])
        if kind=='assent_to_current_proposal':return event('approve')
        if kind=='deny_current_proposal':return event('reject')
        if kind=='pause':return event('pause')
        if kind=='revoke_prior_consent':return event('revoke')
        if kind=='permit_price_lookup':return event('read_permission',**slots)
    if role=='assistant':
        if kind=='propose_cancellation':
            return event('proposal',operation='cancel',reservation_id=slots['reservation'],quoted_refund=slots['amount'],currency=slots['currency'],payment_id=slots['payment'])
        if kind in ('open_help_question','request_user_id','request_reservation_id'):
            return event('question',question=kind)
    return event('unknown',role=role,text=text)


@dataclass
class DecisionState:
    goal: dict | None = None
    identity_claim: dict = field(default_factory=dict)
    proposal: dict | None = None
    consent: str = 'none'
    approved_proposal: str | None = None
    confirmation_open: bool = False
    constraints: dict = field(default_factory=dict)
    read_permissions: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)
    history_guard: list = field(default_factory=list)

    def invalidate(self, status='pending'):
        self.consent=status;self.approved_proposal=None;self.confirmation_open=False

    def apply(self, item: dict):
        kind=item['kind'];data=deepcopy(item['data'])
        if kind=='goal':
            if self.goal is not None and data!=self.goal:
                # "Also do X" versus "replace X" is not inferred from a new goal.
                self.unresolved.append({'reason':'changed_or_additional_goal','prior':self.goal,'new':data})
            if self.proposal:self.history_guard.append({'prior_proposal':self.proposal})
            self.goal=data;self.proposal=None;self.invalidate('none')
        elif kind=='identity_claim':
            if self.identity_claim and self.identity_claim!=data:
                self.unresolved.append({'reason':'identity_changed','old':self.identity_claim,'new':data})
            self.identity_claim=data;self.invalidate('pending' if self.proposal else 'none')
        elif kind=='reservation_reference':
            if not self.goal:self.unresolved.append({'reason':'reservation_without_goal','data':data})
            elif self.goal.get('reservation_id') not in (None,data['reservation']):
                self.unresolved.append({'reason':'reservation_changed','data':data})
            else:self.goal['reservation_id']=data['reservation']
            self.invalidate('pending' if self.proposal else 'none')
        elif kind=='proposal':
            if self.proposal and self.proposal!=data:
                # Keep earlier financial promises: a new quote alone does not
                # prove the user has understood/corrected the old one.
                self.history_guard.append({'prior_proposal':self.proposal})
            self.proposal=data;self.invalidate();self.confirmation_open=True
        elif kind=='approve':
            if self.proposal is None or not self.confirmation_open:
                self.unresolved.append({'reason':'approval_without_proposal'});self.invalidate('unscoped')
            else:
                self.consent='conditional' if self.constraints else 'approved'
                self.approved_proposal=sha256_json(self.proposal)
        elif kind in ('reject','pause','revoke'):
            self.invalidate({'reject':'rejected','pause':'paused','revoke':'revoked'}[kind])
        elif kind=='payment_constraint':
            self.constraints['payment_id']=data['payment_id'];self.invalidate('pending' if self.proposal else 'none')
        elif kind in ('condition','conditional_consent'):
            self.constraints[data['condition']]=True
            self.invalidate('conditional' if kind=='conditional_consent' else 'pending')
        elif kind=='read_permission':
            if data not in self.read_permissions:self.read_permissions.append(data)
            self.invalidate('pending' if self.proposal else 'none')
        elif kind=='question':
            # A fresh question changes the referent of a later "yes".
            self.history_guard.append({'question':data})
            self.invalidate('pending' if self.proposal else 'none')
        elif kind=='unknown':
            self.unresolved.append(data);self.invalidate('unresolved')
        else:raise ValueError(f'unknown event kind: {kind}')

    def payload(self):
        return {'goal':self.goal,'identity_claim':self.identity_claim,'proposal':self.proposal,
                'consent':self.consent,'approved_proposal':self.approved_proposal,'confirmation_open':self.confirmation_open,
                'constraints':self.constraints,'read_permissions':self.read_permissions,
                'history_guard':self.history_guard}


@dataclass(frozen=True)
class StateAssessment:
    comparable: bool
    reasons: tuple[str,...]
    state: dict
    events: tuple[dict,...]

    @property
    def digest(self):return sha256_json(self.state) if self.comparable else None


def assess_messages(messages) -> StateAssessment:
    messages=list(messages);state=DecisionState();events=[];reads={};pending={};tool_reasons=[]
    for index,message in enumerate(messages):
        role=str(_get(message,'role',''));text=text_value(_get(message,'content'))
        if role in ('assistant','user') and text:
            item=parse_event(role,text);state.apply(item);events.append({'message_index':index,**item})
        for call in _get(message,'tool_calls',None) or []:
            if role != 'assistant' or _get(call,'requestor','assistant') != 'assistant':
                tool_reasons.append('unsupported_tool_requestor')
            if state.proposal:state.invalidate('pending')
            function=_get(call,'function',None)
            name=_get(call,'name') or _get(function,'name')
            arguments=_get(call,'arguments',None)
            if arguments is None:arguments=_get(function,'arguments')
            cid=str(_get(call,'id',''))
            if not cid or cid in pending:tool_reasons.append('ambiguous_tool_call')
            pending[cid]={'name':name,'arguments':json_value(arguments)}
        if role=='tool':
            cid=str(_get(message,'id',None) or _get(message,'tool_call_id',''))
            call=pending.pop(cid,None);result=json_value(_get(message,'content'))
            if not call:tool_reasons.append('unmatched_tool_response');continue
            if _get(message,'error',False):tool_reasons.append('tool_error');continue
            if call['name'] in ('get_user_details','get_reservation_details','search_direct_flight','search_onestop_flight','list_all_airports','get_flight_status'):
                key=sha256_json(call)
                if key in reads and reads[key]!=result:state.invalidate('pending' if state.proposal else 'none')
                reads[key]=result
            else:
                tool_reasons.append('write_or_unsupported_tool');state.invalidate('after_write')
    reasons=list(tool_reasons)
    if pending:reasons.append('pending_tools')
    if state.unresolved:reasons.append('unparsed_or_unresolved_history')
    goal=state.goal or {};proposal=state.proposal or {}
    if not goal.get('reservation_id'):reasons.append('incomplete_goal')
    if not proposal:reasons.append('missing_proposal')
    if goal.get('operation') not in ('cancel','change_cabin'):reasons.append('unsupported_goal')
    if proposal and (proposal.get('operation')!=goal.get('operation') or proposal.get('reservation_id')!=goal.get('reservation_id')):
        reasons.append('proposal_goal_mismatch')
    if goal.get('operation')=='change_cabin' and any(proposal.get(k)!=goal.get(k) for k in ('source_cabin','target_cabin')):
        reasons.append('proposal_cabin_mismatch')
    reservation=next((r for r in reads.values() if isinstance(r,dict) and r.get('reservation_id')==goal.get('reservation_id') and 'flights' in r),None)
    user_id=state.identity_claim.get('user')
    user=next((r for r in reads.values() if isinstance(r,dict) and r.get('user_id')==user_id and 'payment_methods' in r),None)
    if not reservation or not user or reservation.get('user_id')!=user_id:
        reasons.append('identity_or_reservation_not_verified')
    else:
        if proposal.get('payment_id') not in user['payment_methods']:reasons.append('unknown_payment_method')
        if goal.get('operation')=='change_cabin':
            if reservation.get('cabin')!=goal.get('source_cabin'):reasons.append('source_cabin_mismatch')
            actual=[{'flight_number':f['flight_number'],'date':f['date']} for f in reservation['flights']]
            if proposal.get('flights')!=actual:reasons.append('flight_scope_mismatch')
    if state.consent not in ('pending','approved','conditional','rejected','paused','revoked'):
        reasons.append('no_scoped_decision')
    # Preserve promises and observed facts separately. Do not replace quoted
    # refund with an amount recomputed from the DB to force equivalence.
    evidence=decision_evidence(messages,version='v2')
    payload={'schema':'decision_state_v4','decision':state.payload(),
             'read_hash':evidence.read_hash,'tool_event_hash':evidence.tool_event_hash}
    return StateAssessment(not reasons,tuple(sorted(set(reasons))),payload,tuple(events))
