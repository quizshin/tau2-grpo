"""Opt-in offline vocabulary: typed identity answers and guarded financial text.

No passenger-set inference from fare arithmetic; no context or price deletion.
Source validation remains the event validator's responsibility.
"""
from copy import deepcopy
import re

from tau3_grpo.algorithms.anchors.semantic_slots import fail, slots
from tau3_grpo.algorithms.anchors import semantic_slots_v2 as v2
from tau3_grpo.utils.hashing import sha256_json

VERSION = 'airline_slots_v3'
ID_CLAUSE = re.compile(
    r'(?:my\s+)?(?P<kind>user|reservation|booking)\s+id\s*(?:is\s+|:\s*)'
    r'(?P<value>[A-Za-z0-9_]+)', re.I)


def identity_answer(text):
    """Recognize only a whole ID-only answer; qualifications stay opaque."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    text = re.sub(r'^(?:sure|certainly|of course)[,.!]?\s*', '', text, flags=re.I)
    values = {}
    while text:
        match = ID_CLAUSE.match(text)
        if not match:
            return None
        key = 'user_id_claim' if match['kind'].lower() == 'user' else 'reservation_id'
        if key in values:
            return None
        values[key] = match['value']
        text = text[match.end():].strip()
        if not text or text in ('.', '!'):
            return values
        separator = re.match(r'(?:,\s*(?:and\s+)?|and\s+)', text, re.I)
        if not separator:
            return None
        text = text[separator.end():].strip()
        if not text:
            return None
    return None


def passenger_scope(value, event, messages):
    text = v2.evidence_text(event)
    if isinstance(value, str) and value.lower() in ('all', 'everyone', 'all passengers'):
        value = {'kind': 'all'}
    if value == {'kind': 'all'} and re.search(
            r'\b(?:not|except|excluding|only|if|unless)\b', text, re.I):
        fail('qualified_all_passengers')
    if value == {'kind': 'all'} and re.search(r'\ball\s+(?:the\s+)?\d+\s+passengers\b', text, re.I):
        # This preserves an explicit universal claim; it does not verify it
        # against a database or convert a stated headcount into a universal.
        return {'kind': 'all'}
    if isinstance(value, dict) and value.get('kind') == 'count':
        count = value.get('count')
        if set(value) != {'kind', 'count'} or type(count) is not int or count < 1:
            fail('invalid_passenger_count')
        if not re.search(r'(?<!\d)' + str(count) + r'\s+passengers\b', text, re.I):
            fail('unsupported_passenger_count')
        at = event.get('at')
        if type(at) is not int or not 0 <= at < len(messages):
            fail('invalid_passenger_count_context')
        # Equal counts are NOT evidence of equal passengers or equal scope.
        return {'kind': 'count', 'count': count,
                'reference_guard': sha256_json(messages[:event['at']+1])}
    return v2.passenger_scope(value, event, messages)


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
    if data.get('operation') == 'other':
        # Retain the existing bounded legacy remove-passenger migration.
        raw = data.get('target')
        if isinstance(raw, dict):
            for field in ('passengers', 'passenger_scope'):
                if field in raw:
                    passenger_scope(raw[field], event, messages)
        v2.operation(data, event, messages)
        return
    data['target'] = target(data.get('target'), event, messages)
    if data['target'].get('action') == data.get('operation'):
        data['target'].pop('action')  # Exact duplicate of the typed operation.
    if data.get('operation') == 'remove_passenger':
        if 'passenger_scope' not in data['target'] or not re.search(
                r'\b(?:remove|delete)\b', v2.evidence_text(event), re.I):
            fail('unsupported_remove_passenger')


def question(data, event):
    if set(data) not in ({'intent', 'proposal_id'}, {'topic', 'proposal_id'}):
        fail('invalid_question_fields')
    intent = data.get('intent', data.get('topic'))
    text = v2.evidence_text(event)
    if intent in ('explore_options', 'choose_option'):
        if data['proposal_id'] is not None:
            fail('options_question_is_not_approval')
        pattern = r'\b(?:explore|options?|which|prefer)\b'
        if not re.search(pattern, text, re.I):
            fail('unsupported_options_question')
        return {'topic': intent + ':' + sha256_json(text), 'proposal_id': None}
    if intent in ('identity_user_id', 'identity_reservation_id', 'identity_details'):
        fields = [key for pattern, key in [(r'\buser id\b', 'user_id_claim'),
                   (r'\b(?:reservation|booking) id\b', 'reservation_id')]
                  if re.search(pattern, text, re.I)]
        if not fields or data['proposal_id'] is not None or not re.search(
                r'\b(?:provide|tell|share|what|may i have|could i have)\b', text, re.I):
            fail('unsupported_identity_question')
        # Preserve other clauses in the question; field overlap alone must not
        # erase a restriction, a requested document, or an extra operation.
        return {'topic': 'identity:' + ','.join(sorted(fields)) + ':' + sha256_json(text),
                'proposal_id': None}
    if data['proposal_id'] is not None and re.search(
            r'\b(?:which|options?|user id|reservation id|booking id|passport|search)\b'
            r'|\b(?:provide|choose|select|confirm)\s+(?:your |a |the )?payment method\b',
            text, re.I):
        fail('confirmation_question_scope_conflict')
    return v2.question(data, event)


def normalize_packet(packet, messages):
    packet = deepcopy(packet)
    if not isinstance(packet, dict) or not isinstance(packet.get('events'), list):
        fail('invalid_slot_packet')
    for event in packet['events']:
        if not isinstance(event, dict) or not isinstance(event.get('data'), dict):
            fail('invalid_slot_event')
        data, kind = event['data'], event.get('kind')
        if kind == 'identity':
            at = event.get('at')
            if type(at) is not int or not 0 <= at < len(messages):
                fail('invalid_identity_message')
            expected = identity_answer(messages[at].get('content'))
            if expected is None or data != {'values': expected}:
                fail('unsupported_identity_answer')
        elif kind == 'goal':
            at = event.get('at')
            if type(at) is int and 0 <= at < len(messages) and identity_answer(messages[at].get('content')):
                fail('identity_answer_must_not_change_goal')
            operation(data, event, messages)
        elif kind == 'proposal':
            if not isinstance(data.get('operations'), list):
                fail('invalid_slot_operations')
            for op in data['operations']:
                if not isinstance(op, dict):
                    fail('invalid_slot_operation')
                operation(op, event, messages)
                op['terms'] = target(op.get('terms'), event, messages, terms=True)
        elif kind == 'constraint':
            data['predicate'] = v2.fee_predicate(data.get('predicate'), event)
        elif kind == 'question':
            event['data'] = question(data, event)
    return packet


def check_money_roles(messages, event, terms):
    """Current assistant amounts only; Markdown labels do not hide their role."""
    from tau3_grpo.algorithms.anchors.grounded import text_claims
    at = event['at']; message = messages[at]; text = message['content']
    supported = {'quoted_refund': [], 'quoted_charge': []}
    for claim in text_claims(message, at):
        if claim['kind'] != 'money_mention':
            continue
        start, end = claim['evidence']['start'], claim['evidence']['end']
        if not any(e.get('message_index') == at and 'start' in e
                   and e['start'] <= start < end <= e['end'] for e in event['evidence']):
            continue
        before = re.sub(r'[*_]', '', text[max(0, start-65):start]).lower()
        after = re.sub(r'[*_]', '', text[end:end+120]).lower()
        if (re.search(r'\b(?:not|no|never|without)\b', before)
                or re.match(r'\s*[-+×*/=]', after)):
            continue
        refund = bool(re.search(r'\brefund(?:ing| amount)?(?: of| is|:)?\s*$', before)
                      or re.match(r'\s*(?:refund|back)\b', after))
        charge = bool(re.search(r'\b(?:charge|charging|pay|fee)(?: of| is|:)?\s*$', before)
                      or re.match(r'\s*(?:charge|fee)\b', after))
        # A bounded Markdown table row with one value (optionally preceded by
        # an empty/"-" old-value cell). Two monetary columns stay unsupported.
        row_before = before.rsplit('\n', 1)[-1]
        row_after = after.split('\n', 1)[0]
        if re.fullmatch(r'\s*\|\s*', row_after):
            refund |= bool(re.fullmatch(r'\s*\|\s*refund(?: amount)?\s*\|\s*(?:-\s*\|\s*)?', row_before))
            charge |= bool(re.fullmatch(r'\s*\|\s*charge(?: amount)?\s*\|\s*(?:-\s*\|\s*)?', row_before))
        if re.search(r'\btotal price difference:\s*$', before) and re.match(
                r'\s*[.!]?\s*this (?:would|will) be charged\b', after):
            charge = True
        if refund != charge:
            supported['quoted_refund' if refund else 'quoted_charge'].append(claim['value'])
    for key, values in supported.items():
        if key in terms and {'currency': terms.get('currency'), 'amount': terms[key]} not in values:
            fail('unsupported_money_role:' + key)


def financial_context(messages):
    """Retain every financial assistant statement, including table/reference prices.

    Exact text deliberately prevents broad paraphrase merges. A number in a
    fare table is not automatically a charge or a refund, nor permission to pay.
    """
    from tau3_grpo.algorithms.anchors.grounded import text_claims
    result = []
    for index, message in enumerate(messages):
        if message.get('role') != 'assistant' or not isinstance(message.get('content'), str):
            continue
        amounts = [c['value'] for c in text_claims(message, index) if c['kind'] == 'money_mention']
        if amounts:
            result.append({'text': message['content'], 'amounts': amounts})
    return result
