"""Versioned, conservative Airline slot vocabulary for offline semantic audits."""
import re
from copy import deepcopy
from decimal import Decimal

VERSION = 'airline_slots_v1'
TARGET_ALIASES = {'booking_id': 'reservation_id'}
TERM_ALIASES = {'refund_amount': 'quoted_refund', 'charge_amount': 'quoted_charge',
                'refund': 'quoted_refund', 'charge': 'quoted_charge',
                'refund_currency': 'currency', 'charge_currency': 'currency'}
TARGET_FIELDS = {'reservation_id', 'user_id_claim', 'action', 'request', 'information',
                 'source_cabin', 'target_cabin', 'passengers', 'date', 'add_bags',
                 'flight_type', 'flight_ids', 'origin', 'destination'}
TERM_FIELDS = {'quoted_refund', 'quoted_charge', 'currency', 'refund_payment_id',
               'charge_payment_id', 'refund_timing', 'fee_condition', 'insurance',
               'passengers', 'flight_ids', 'date', 'add_bags'}


def fail(reason):
    from tau3_grpo.algorithms.anchors.semantic_state import SemanticError
    raise SemanticError(reason)


def slots(value, *, terms=False):
    if not isinstance(value, dict):
        fail('slot_object_required')
    aliases = TERM_ALIASES if terms else TARGET_ALIASES
    allowed = TERM_FIELDS if terms else TARGET_FIELDS
    result = {}
    for raw, child in value.items():
        key = aliases.get(raw, raw)
        if key not in allowed:
            fail('unsupported_slot:' + str(raw))
        if key in ('quoted_refund', 'quoted_charge'):
            if type(child) not in (str, int) or not re.fullmatch(r'\d+(?:\.\d+)?', str(child)):
                fail('invalid_money_slot')
            child = format(Decimal(str(child)).normalize(), 'f')
        elif key == 'currency':
            if not isinstance(child, str) or not re.fullmatch('[A-Za-z]{3}', child):
                fail('invalid_currency_slot')
            child = child.upper()
        elif key == 'add_bags':
            if type(child) is not int or child < 0:
                fail('invalid_baggage_slot')
        elif key in ('passengers', 'flight_ids'):
            if not (isinstance(child, str) and child or isinstance(child, list)
                    and child and all(isinstance(x, str) and x for x in child)):
                fail('invalid_scope_slot')
            # Retain list order: do not assume every scope is an unordered set.
        elif child is not None and (not isinstance(child, str) or not child):
            fail('invalid_text_slot:' + key)
        if key in ('source_cabin', 'target_cabin') and isinstance(child, str):
            child = {'business class': 'business', 'economy class': 'economy'}.get(child.lower(), child)
        if key in result and result[key] != child:
            fail('conflicting_slot_alias:' + key)
        result[key] = child
    if terms and any(k in result for k in ('quoted_refund', 'quoted_charge')) and 'currency' not in result:
        # Unknown currency is not silently supplied from DB or a default locale.
        fail('missing_money_currency')
    return result


def normalize_packet(packet, version, *, messages=None):
    if version == 'airline_slots_v4':
        from tau3_grpo.algorithms.anchors.semantic_slots_v4 import normalize_packet as v4
        return v4(packet, messages if messages is not None else [])
    if version == 'airline_slots_v3':
        from tau3_grpo.algorithms.anchors.semantic_slots_v3 import normalize_packet as v3
        return v3(packet, messages if messages is not None else [])
    if version == 'airline_slots_v2':
        from tau3_grpo.algorithms.anchors.semantic_slots_v2 import normalize_packet as v2
        return v2(packet, messages if messages is not None else [])
    if version != VERSION:
        fail('unsupported_slot_schema')
    normalized = deepcopy(packet)
    if not isinstance(normalized, dict) or not isinstance(normalized.get('events'), list):
        fail('invalid_slot_packet')
    for event in normalized['events']:
        if not isinstance(event, dict) or not isinstance(event.get('data'), dict):
            fail('invalid_slot_event')
        data = event['data']
        if event.get('kind') == 'goal':
            data['target'] = slots(data.get('target'))
        elif event.get('kind') == 'proposal':
            if not isinstance(data.get('operations'), list):
                fail('invalid_slot_operations')
            for operation in data['operations']:
                if not isinstance(operation, dict):
                    fail('invalid_slot_operation')
                operation['target'] = slots(operation.get('target'))
                operation['terms'] = slots(operation.get('terms'), terms=True)
        elif event.get('kind') == 'constraint':
            predicate = data.get('predicate')
            if not isinstance(predicate, dict) or set(predicate) != {'fee'}:
                fail('unsupported_constraint_predicate')
            fee = predicate['fee']
            if (not isinstance(fee, dict) or set(fee) != {'op', 'amount', 'currency'}
                    or fee['op'] not in ('eq', 'lte', 'lt', 'gte', 'gt')):
                fail('unsupported_fee_predicate')
            money = slots({'quoted_charge': fee['amount'], 'currency': fee['currency']}, terms=True)
            data['predicate'] = {'fee': {'op': fee['op'], 'amount': money['quoted_charge'],
                                         'currency': money['currency']}}
    return normalized


def check_money_roles(messages, event, terms):
    """Require a current, cited assistant amount with an explicit financial role.

    This bounded lexical check deliberately abstains on more complex phrasing.
    It does not establish complete semantic correctness or tool authorization.
    """
    from tau3_grpo.algorithms.anchors.grounded import text_claims
    at = event['at']
    message = messages[at]
    text = message['content']
    supported = {'quoted_refund': [], 'quoted_charge': []}
    for claim in text_claims(message, at):
        if claim['kind'] != 'money_mention':
            continue
        start, end = claim['evidence']['start'], claim['evidence']['end']
        if not any(e.get('message_index') == at and 'start' in e
                   and e['start'] <= start < end <= e['end'] for e in event['evidence']):
            continue
        before, after = text[max(0, start-50):start].lower(), text[end:end+40].lower()
        refund = bool(re.search(r'(?:refund|refunding)(?: of| is|:)?\s*$', before)
                      or re.match(r'\s*(?:refund|back)\b', after))
        charge = bool(re.search(r'(?:charge|charging|pay|fee)(?: of| is|:)?\s*$', before)
                      or re.match(r'\s*(?:charge|fee)\b', after))
        if refund != charge:
            supported['quoted_refund' if refund else 'quoted_charge'].append(claim['value'])
    for field in supported:
        if field in terms and {'currency': terms.get('currency'), 'amount': terms[field]} not in supported[field]:
            fail('unsupported_money_role:' + field)


def check_proposal_money_coverage(messages, event):
    """Do not accept a quote that simply omits a visible monetary amount."""
    from tau3_grpo.algorithms.anchors.grounded import text_claims
    represented = []
    for operation in event['data']['operations']:
        terms = operation['terms']
        represented.extend({'currency': terms.get('currency'), 'amount': terms[field]}
                           for field in ('quoted_refund', 'quoted_charge') if field in terms)
    for claim in text_claims(messages[event['at']], event['at']):
        if claim['kind'] == 'money_mention' and claim['value'] not in represented:
            fail('unrepresented_communicated_money')


def check_identity_evidence(event, values):
    """Known identifiers must occur in the event's already-validated evidence."""
    for key in ('reservation_id', 'user_id_claim', 'refund_payment_id', 'charge_payment_id'):
        value = values.get(key)
        if not isinstance(value, str):
            continue
        found = any((isinstance(e.get('quote'), str)
                     and re.search(r'(?<!\w)' + re.escape(value) + r'(?!\w)', e['quote']))
                    or e.get('value') == value for e in event['evidence'])
        if not found:
            fail('unsupported_identity_slot:' + key)
