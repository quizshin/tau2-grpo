"""Offline evidence extraction, NOT an authorization or semantic-equivalence model.

Surface mentions and observed facts have different types. Every item is rooted
in an exact visible message. Unsupported language remains in an exact guard;
extracting a few slots never makes a prefix semantically complete.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from decimal import Decimal
import json
import re

from tau3_grpo.utils.hashing import sha256_json
from tau3_grpo.algorithms.anchors.decision_state import parse_event

SCHEMA = 'grounded_evidence_v1'
# These patterns locate mentions, not affirmative goals, promises or consent.
PATTERNS = {
    'user_id_mention': r'\b[a-z][a-z_]+_\d+\b',
    'reservation_id_mention': r'\b[A-Z0-9]{6}\b',
    'payment_id_mention': r'\b(?:credit_card|gift_card|certificate)_\d+\b',
    'card_tail_mention': r'(?i:ending in|ends in)\s*\*{0,2}(?P<value>\d{4})\b',
    'flight_mention': r'\bHAT\d{3}\b',
    'cabin_mention': r'(?i:\bbasic economy\b|\beconomy\b|\bbusiness(?: class)?\b)',
    'money_mention': r'(?P<sign>-)?(?P<currency>USD|EUR|GBP|\$|€|£)\s*(?P<amount>-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\b',
    'date_mention': r'\b\d{4}-\d{2}-\d{2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}(?:, \d{4})?\b',
    'qualification_marker': r"(?i:\bif\b|\bunless\b|\bonly\b|\bnot\b|\bno\b|\bbefore\b|\bbut\b|\bdon['’]t\b)",
}
READ_TOOLS = {'get_user_details', 'get_reservation_details', 'search_direct_flight',
              'search_onestop_flight', 'get_flight_status', 'list_all_airports'}


def _content(message):
    value = message.get('content')
    return value if isinstance(value, str) else ''


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


def _canonical(kind, match):
    if kind == 'money_mention':
        currency = {'$': 'USD', '€': 'EUR', '£': 'GBP'}.get(match['currency'], match['currency'])
        amount = Decimal(match['amount'].replace(',', ''))
        if match['sign']:
            amount = -amount
        return {'currency': currency, 'amount': format(amount.normalize(), 'f')}
    if kind == 'card_tail_mention':
        return match['value']
    if kind == 'cabin_mention':
        return match[0].lower().replace(' class', '').replace(' ', '_')
    return match[0]


def text_claims(message, index):
    """Each claim is only a surface occurrence, including negated mentions."""
    role = message.get('role')
    if role not in ('user', 'assistant'):
        return []
    text = _content(message)
    items = []
    for kind, pattern in PATTERNS.items():
        for match in re.finditer(pattern, text):
            value = _canonical(kind, match)
            if kind == 'user_id_mention' and re.fullmatch(PATTERNS['payment_id_mention'], value):
                continue
            if kind == 'reservation_id_mention' and (not any(c.isalpha() for c in value)
                                                      or not any(c.isdigit() for c in value)
                                                      or re.fullmatch(PATTERNS['flight_mention'], value)):
                continue
            items.append({'kind': kind, 'value': value, 'epistemic': role + '_mention',
                          'evidence': {'message_index': index, 'role': role, 'start': match.start(),
                                       'end': match.end(), 'quote': match[0],
                                       'message_sha256': sha256_json(message)}})
    return sorted(items, key=lambda item: (item['evidence']['start'], item['kind']))


def _leaves(value, pointer=''):
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            escaped = key.replace('~', '~0').replace('/', '~1')
            yield from _leaves(child, pointer + '/' + escaped)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _leaves(child, pointer + '/' + str(index))
    else:
        yield pointer, value


def _pointer(value, pointer):
    for token in pointer.split('/')[1:]:
        token = token.replace('~1', '/').replace('~0', '~')
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def extract(messages):
    """Extract from a visible pre-action prefix only; no DB/task/reward input.

    Tool observations are linked by ordered call IDs. Local evidence validation
    is separate from external DB replay certification. Full original messages
    remain in the guard, including errors, unknown clauses and previous quotes.
    """
    messages = deepcopy(list(messages))
    claims, observations, issues = [], [], []
    pending, used = deque(), set()
    for index, message in enumerate(messages):
        role = message.get('role')
        claims.extend(text_claims(message, index))
        if pending and role != 'tool':
            issues.append({'kind': 'missing_tool_response', 'message_index': index})
        calls = message.get('tool_calls') or []
        for call in calls:
            cid = call.get('id')
            if role != 'assistant' or call.get('requestor', 'assistant') != 'assistant':
                issues.append({'kind': 'unsupported_requestor', 'message_index': index})
            if not cid or cid in used:
                issues.append({'kind': 'duplicate_or_missing_call_id', 'message_index': index})
            used.add(cid)
            pending.append((index, call))
        if role != 'tool':
            continue
        cid = message.get('id') or message.get('tool_call_id')
        if not pending or pending[0][1].get('id') != cid:
            issues.append({'kind': 'unmatched_tool_response', 'message_index': index})
            continue
        call_index, call = pending.popleft()
        if (messages[call_index].get('role') != 'assistant'
                or call.get('requestor', 'assistant') != 'assistant'):
            continue
        if message.get('requestor', 'assistant') != 'assistant':
            issues.append({'kind': 'unsupported_response_requestor', 'message_index': index})
            continue
        if message.get('error', False):
            issues.append({'kind': 'tool_error', 'message_index': index})
            continue
        function = call.get('function') or {}
        name = call.get('name') or function.get('name')
        try:
            result = _json(message.get('content'))
            arguments = _json(call.get('arguments', function.get('arguments', {})))
        except (TypeError, ValueError):
            issues.append({'kind': 'invalid_tool_json', 'message_index': index})
            continue
        if name not in READ_TOOLS:
            issues.append({'kind': 'write_or_unsupported_tool', 'message_index': index})
            continue
        # Prevent a mismatched tool object from verifying a different lookup.
        expected = {'get_user_details': 'user_id', 'get_reservation_details': 'reservation_id'}.get(name)
        if expected and (not isinstance(result, dict) or not isinstance(arguments, dict)
                         or not arguments.get(expected) or result.get(expected) != arguments[expected]):
            issues.append({'kind': 'lookup_identity_mismatch', 'message_index': index})
            continue
        observations.append({'tool_name': name, 'arguments': arguments, 'result': result,
                             'call_message_index': call_index, 'message_index': index, 'call_id': cid})
    if pending:
        issues.append({'kind': 'pending_tools', 'message_index': len(messages)})

    facts = []
    for observation in observations:
        for pointer, value in _leaves(observation['result']):
            facts.append({'kind': 'observed_tool_value', 'value': value, 'epistemic': 'tool_observation',
                          'tool_name': observation['tool_name'],
                          'evidence': {'message_index': observation['message_index'], 'role': 'tool',
                                       'call_message_index': observation['call_message_index'],
                                       'call_id': observation['call_id'], 'pointer': pointer,
                                       'message_sha256': sha256_json(messages[observation['message_index']])}})
    links = []
    for ci, claim in enumerate(claims):
        kind, value = claim['kind'], claim['value']
        if kind not in ('user_id_mention', 'reservation_id_mention', 'payment_id_mention', 'card_tail_mention'):
            continue
        candidates = []
        for fi, fact in enumerate(facts):
            pointer = fact['evidence']['pointer']
            valid_path = ((kind == 'user_id_mention' and pointer == '/user_id')
                          or (kind == 'reservation_id_mention' and pointer == '/reservation_id')
                          or (kind == 'payment_id_mention' and pointer.startswith('/payment_methods/') and pointer.endswith('/id'))
                          or (kind == 'card_tail_mention' and pointer.startswith('/payment_methods/') and pointer.endswith('/last_four')))
            if valid_path and value == fact['value']:
                candidates.append(fi)
        if candidates:
            # A tail shared by two cards is unresolved; no arbitrary first match.
            identities = set()
            for fi in candidates:
                fact = facts[fi]
                observation = next(o for o in observations if o['message_index'] == fact['evidence']['message_index'])
                if kind == 'card_tail_mention':
                    parts = fact['evidence']['pointer'].split('/')
                    identities.add((observation['result'].get('user_id'), parts[2]))
                else:
                    identities.add((observation['result'].get('user_id'), str(value)))
            links.append({'claim': ci, 'facts': candidates, 'status': 'matched' if len(identities) == 1 else 'ambiguous',
                          'meaning': 'value_correspondence_only_not_ownership_intent_or_consent'})

    text_guard = [{'role': m.get('role'), 'content': _content(m)} for m in messages
                  if m.get('role') in ('assistant', 'user') and _content(m)]
    # Separate exact text protection from structural fields. NEVER use only the
    # latter as a training key: rejected options also have matching mentions.
    result = {'schema': SCHEMA, 'prefix_sha256': sha256_json(messages), 'claims': claims,
              'facts': facts, 'links': links, 'issues': issues,
              'guard': text_guard, 'guard_sha256': sha256_json(text_guard),
              'semantic_complete': False, 'training_eligible': False}
    result['decision_view'] = decision_view(messages, claims, observations, links)
    return result


def decision_view(messages, claims, observations, links):
    """Evidence-backed review view. Candidates are explicitly not approvals.

    Keep all requests, earlier quotes and qualifications instead of overwriting
    with the last ID/amount. DB ownership is distinct from speaker identity.
    """
    associations = []
    users = [o for o in observations if o['tool_name'] == 'get_user_details']
    reservations = [o for o in observations if o['tool_name'] == 'get_reservation_details']
    for reservation in reservations:
        for user in users:
            if reservation['result'].get('user_id') == user['result']['user_id']:
                associations.append({'user_id': user['result']['user_id'],
                                     'reservation_id': reservation['result']['reservation_id'],
                                     'user_message_index': user['message_index'],
                                     'reservation_message_index': reservation['message_index'],
                                     'meaning': 'observed_db_ownership_not_speaker_authentication'})
    quotes = []
    for index, message in enumerate(messages):
        if message.get('role') != 'assistant':
            continue
        money = [i for i, c in enumerate(claims) if c['kind'] == 'money_mention'
                 and c['evidence']['message_index'] == index]
        if money:
            quotes.append({'message_index': index, 'text': _content(message), 'money_claims': money,
                           'meaning': 'communicated_numbers_not_verified_refund'})
    responses = []
    for index, message in enumerate(messages):
        if message.get('role') != 'user' or not _content(message):
            continue
        item = parse_event('user', _content(message))
        if item['kind'] in ('approve', 'reject', 'pause', 'revoke', 'conditional_consent'):
            # Adjacent assistant text is a reference for review, not proof of a
            # complete proposal. "Yes" to a search question is not write consent.
            adjacent = index - 1 if index and messages[index-1].get('role') == 'assistant' else None
            responses.append({'message_index': index, 'phrase_kind': item['kind'],
                              'responding_to_message_index': adjacent,
                              'approved_operation': None, 'requires_scope_review': True})
    return {'user_context': [{'message_index': i, 'text': _content(m)} for i, m in enumerate(messages)
                             if m.get('role') == 'user' and _content(m)],
            'observed_associations': associations, 'communicated_quotes': quotes,
            'payment_correspondences': [link for link in links if claims[link['claim']]['kind']
                                       in ('payment_id_mention', 'card_tail_mention')],
            'response_candidates': responses, 'goal_resolution': 'requires_semantic_review',
            'condition_resolution': 'full_user_text_retained',
            'proposal_resolution': 'full_quote_history_retained'}


def validate(messages, report):
    """Fail closed on stale/future/altered evidence or missing protective text.

    Re-extraction also enforces completeness for this *bounded lexical schema*,
    never completeness of the natural-language decision state.
    """
    messages = list(messages)
    errors = []
    if report.get('schema') != SCHEMA or report.get('prefix_sha256') != sha256_json(messages):
        errors.append('schema_or_prefix_mismatch')
    for item in report.get('claims', []) + report.get('facts', []):
        try:
            ev = item['evidence']; index = ev['message_index']
            if not isinstance(index, int) or index < 0 or index >= len(messages):
                raise ValueError('outside_prefix')
            message = messages[index]
            if ev['role'] != message.get('role') or ev['message_sha256'] != sha256_json(message):
                raise ValueError('source_mismatch')
            if item['epistemic'] == 'tool_observation':
                if message.get('role') != 'tool' or message.get('error', False):
                    raise ValueError('invalid_observation')
                if _pointer(_json(message.get('content')), ev['pointer']) != item['value']:
                    raise ValueError('value_mismatch')
            else:
                text = _content(message); start, end = ev['start'], ev['end']
                if not (0 <= start < end <= len(text)) or text[start:end] != ev['quote']:
                    raise ValueError('span_mismatch')
                if item not in text_claims(message, index):
                    raise ValueError('unsupported_interpretation')
        except (KeyError, IndexError, TypeError, ValueError):
            errors.append('invalid_evidence')
    # Python equality equates True with 1 and False with 0. Evidence must keep
    # JSON scalar types as well as values, so compare canonical representations.
    if sha256_json(report) != sha256_json(extract(messages)):
        errors.append('extraction_or_guard_mismatch')
    return sorted(set(errors))
