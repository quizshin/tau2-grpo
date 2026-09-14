"""Typed fee predicates, passenger selectors and questions; no model calls."""
import re
from copy import deepcopy

from tau3_grpo.algorithms.anchors.semantic_slots import fail, slots
from tau3_grpo.utils.hashing import sha256_json

VERSION = 'airline_slots_v2'
FEE_SCOPES = {'all', 'change', 'cancellation', 'service'}


def evidence_text(event):
    return ' '.join(e['quote'] for e in event.get('evidence', [])
                    if isinstance(e, dict) and isinstance(e.get('quote'), str)
                    and e.get('message_index') == event.get('at'))


def no_fee_scopes(text):
    return {m.group(1) or 'all' for m in re.finditer(
        r'\b(?:no|without)\s+(?:(change|cancellation|service)\s+)?fees?\b', text.lower())}


def fee_predicate(value, event):
    if not isinstance(value, dict) or set(value) != {'fee'} or not isinstance(value['fee'], dict):
        fail('unsupported_constraint_predicate')
    fee = deepcopy(value['fee'])
    text = evidence_text(event)
    # Migrate a legacy *unknown-currency* zero only with explicit no-fee text.
    # A stated USD zero is not erased or generalized to every currency.
    if set(fee) == {'op', 'amount', 'currency'}:
        found = no_fee_scopes(text)
        if fee['op'] == 'eq' and str(fee['amount']) in ('0', '0.0', '0.00') and fee['currency'] is None:
            if len(found) != 1:
                fail('ambiguous_no_fee_scope')
            fee = {'kind': 'none', 'scope': next(iter(found))}
        else:
            fee = {'kind': 'amount_bound', 'scope': 'all', **fee}
    if fee.get('scope') not in FEE_SCOPES:
        fail('invalid_fee_scope')
    if fee.get('kind') == 'none':
        if set(fee) != {'kind', 'scope'} or no_fee_scopes(text) != {fee['scope']}:
            fail('unsupported_no_fee_scope')
    elif fee.get('kind') == 'amount_bound':
        if set(fee) != {'kind', 'scope', 'op', 'amount', 'currency'} or fee['op'] not in ('eq', 'lte', 'lt', 'gte', 'gt'):
            fail('invalid_fee_bound')
        money = slots({'quoted_charge': fee['amount'], 'currency': fee['currency']}, terms=True)
        fee.update(amount=money['quoted_charge'], currency=money['currency'])
        from tau3_grpo.algorithms.anchors.grounded import text_claims
        supported = [c['value'] for c in text_claims({'role': 'user', 'content': text}, 0)
                     if c['kind'] == 'money_mention']
        if {'amount': fee['amount'], 'currency': fee['currency']} not in supported:
            fail('unsupported_fee_bound_money')
        scoped = set(re.findall(r'\b(change|cancellation|service) fees?\b', text.lower()))
        if (fee['scope'] == 'all' and scoped or fee['scope'] != 'all' and scoped != {fee['scope']}):
            fail('unsupported_fee_bound_scope')
    else:
        fail('invalid_fee_kind')
    return {'fee': fee}


def passenger_scope(value, event, messages):
    text = evidence_text(event)
    if isinstance(value, str):
        if value.lower() in ('all', 'everyone', 'all passengers'):
            value = {'kind': 'all'}
        elif value.lower() in ('second passenger', 'the second passenger'):
            value = {'kind': 'ordinal', 'position': 2}
        else:
            fail('ambiguous_legacy_passenger_scope')
    if not isinstance(value, dict):
        fail('invalid_passenger_scope')
    value = deepcopy(value)
    kind = value.get('kind')
    if kind == 'all':
        if set(value) != {'kind'} or not re.search(r'\b(?:everyone|all(?: the)? passengers|everyone on|downgrade everyone)\b', text, re.I):
            fail('unsupported_all_passengers')
    elif kind == 'ordinal':
        if set(value) != {'kind', 'position'} or type(value['position']) is not int or value['position'] <= 0:
            fail('invalid_passenger_ordinal')
        n = value['position']
        word = {1: 'first', 2: 'second', 3: 'third'}.get(n, f'{n}(?:st|nd|rd|th)')
        if not re.search(r'\b(?:' + word + r') passenger\b', text, re.I):
            fail('unsupported_passenger_ordinal')
        at = event['at']
        if type(at) is not int or at < 0 or at >= len(messages):
            fail('invalid_ordinal_context')
        # This is an unresolved ordinal, NOT a verified passenger ID. Separate
        # contexts never merge merely because both mention position two.
        value['reference_guard'] = sha256_json(messages[:at+1])
    elif kind == 'ids':
        ids = value.get('values')
        if set(value) != {'kind', 'values'} or not isinstance(ids, list) or not ids or not all(isinstance(v, str) and v for v in ids):
            fail('invalid_passenger_ids')
        for item in ids:
            if not any((isinstance(e.get('quote'), str) and re.search(
                    r'\bpassenger(?: id)?\s*[:#]?\s*' + re.escape(item) + r'(?!\w)', e['quote'], re.I))
                       or (e.get('value') == item and isinstance(e.get('pointer'), str)
                           and re.search(r'(?:^|/)passengers/\d+/(?:id|passenger_id)$', e['pointer']))
                       for e in event['evidence']):
                fail('unsupported_passenger_id')
        value['values'] = sorted(set(ids))
    else:
        fail('invalid_passenger_scope_kind')
    return value


def target(value, event, messages, *, terms=False):
    if not isinstance(value, dict):
        fail('slot_object_required')
    value = deepcopy(value)
    selectors = [passenger_scope(value.pop(k), event, messages)
                 for k in ('passengers', 'passenger_scope') if k in value]
    if len(selectors) == 2 and selectors[0] != selectors[1]:
        fail('conflicting_passenger_scope')
    result = slots(value, terms=terms)
    if selectors:
        result['passenger_scope'] = selectors[0]
    return result


def operation(data, event, messages):
    raw = deepcopy(data.get('target'))
    if not isinstance(raw, dict):
        fail('slot_object_required')
    name = data.get('operation')
    if name == 'other' and raw.get('action') == 'remove_second_passenger':
        raw.pop('action')
        implicit = passenger_scope({'kind': 'ordinal', 'position': 2}, event, messages)
        normalized = target(raw, event, messages)
        if 'passenger_scope' in normalized and normalized['passenger_scope'] != implicit:
            fail('conflicting_passenger_scope')
        normalized['passenger_scope'] = implicit
        name = 'remove_passenger'
    else:
        if name == 'other' and raw.get('action') == 'remove' and ('passengers' in raw or 'passenger_scope' in raw):
            raw.pop('action')
            name = 'remove_passenger'
        normalized = target(raw, event, messages)
    if name == 'remove_passenger':
        if 'passenger_scope' not in normalized or not re.search(r'\b(?:remove|delete)\b', evidence_text(event), re.I):
            fail('unsupported_remove_passenger')
    data['operation'], data['target'] = name, normalized


def question(data, event):
    if set(data) not in ({'topic', 'proposal_id'}, {'intent', 'proposal_id'}):
        fail('invalid_question_fields')
    name = data.get('intent', data.get('topic'))
    pid = data['proposal_id']
    text = evidence_text(event)
    if not isinstance(name, str) or not name:
        fail('invalid_question_intent')
    if 'intent' in data and name not in ('confirm_proposal', 'identity_user_id', 'search_permission'):
        fail('unsupported_question_intent')
    if pid is not None:
        # Only known confirmation descriptions may migrate. A proposal binding
        # by itself does not authorize interpreting an identity/search question.
        if not isinstance(pid, str) or not name.startswith('confirm') or not re.search(
                r'\bconfirm\b|\bdo you approve\b|\bshall i proceed\b', text, re.I):
            fail('unsupported_confirmation_question')
        if re.search(r'\b(?:user id|identity|passport|search)\b', text, re.I):
            fail('confirmation_question_scope_conflict')
        topic = 'confirm_proposal'
    elif name in ('confirm_proposal',) or name.startswith('confirm '):
        fail('unbound_confirmation_question')
    elif name in ('identity_user_id', 'identity_question', 'user_id'):
        if not re.search(r'\buser id\b', text, re.I):
            fail('unsupported_identity_question')
        topic = 'identity_user_id'
    elif name == 'search_permission':
        if not re.search(r'\bsearch\b', text, re.I):
            fail('unsupported_search_question')
        topic = 'search_permission:' + sha256_json(text)
    elif 'intent' in data:
        fail('unsupported_question_intent')
    else:
        # Legacy unknown topics remain exact guards, not broad equivalence.
        topic = 'unclassified:' + sha256_json([name, text])
    return {'topic': topic, 'proposal_id': pid}


def normalize_packet(packet, messages):
    packet = deepcopy(packet)
    if not isinstance(packet, dict) or not isinstance(packet.get('events'), list):
        fail('invalid_slot_packet')
    for event in packet['events']:
        if not isinstance(event, dict) or not isinstance(event.get('data'), dict):
            fail('invalid_slot_event')
        data = event['data']
        if event.get('kind') == 'goal':
            operation(data, event, messages)
        elif event.get('kind') == 'proposal':
            if not isinstance(data.get('operations'), list):
                fail('invalid_slot_operations')
            for op in data['operations']:
                if not isinstance(op, dict):
                    fail('invalid_slot_operation')
                operation(op, event, messages)
                op['terms'] = target(op.get('terms'), event, messages, terms=True)
        elif event.get('kind') == 'constraint':
            data['predicate'] = fee_predicate(data.get('predicate'), event)
        elif event.get('kind') == 'question':
            event['data'] = question(data, event)
    return packet
