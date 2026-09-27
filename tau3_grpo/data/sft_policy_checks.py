"""Narrow evidence-based checks for SFT; these do not certify task success."""
from __future__ import annotations

import json

from tau3_grpo.prompts import build_system_prompt
from tau3_grpo.utils.hashing import sha256_text

POLICY_SHA256 = 'c0e0f60839a5d0c5c2cf76f6625cc84ce89392a3fcab93ad1696a1a317d3d9e3'

ALLOWANCE = {
    'regular': {'basic_economy': 0, 'economy': 1, 'business': 2},
    'silver': {'basic_economy': 1, 'economy': 2, 'business': 3},
    'gold': {'basic_economy': 2, 'economy': 3, 'business': 4},
}


def audit_baggage_allowances(messages):
    """Read only prior observations; never use a later receipt to justify arguments.

    Checks the count of paid bags, not ambiguous natural-language user quantities.
    Existing paid-bag inconsistencies and unknown membership are not guessed away.
    """
    policy = build_system_prompt()
    if (sha256_text(policy) != POLICY_SHA256 or not messages
            or messages[0].get('content') != policy):
        raise ValueError('Policy identity changed; review deterministic baggage rule')
    users, reservations, checks = {}, {}, []
    for index, message in enumerate(messages):
        if message['role'] == 'tool':
            try:
                payload = json.loads(message.get('content') or '')
            except (ValueError, TypeError):
                continue
            if not isinstance(payload, dict):
                continue
            if message.get('name') == 'get_user_details' and payload.get('user_id'):
                users[payload['user_id']] = payload
            if payload.get('reservation_id') and 'passengers' in payload and 'cabin' in payload:
                reservations[payload['reservation_id']] = payload
        if message['role'] != 'assistant':
            continue
        for call in message.get('tool_calls') or []:
            function = call.get('function', call)
            name = function.get('name')
            if name not in ('book_reservation', 'update_reservation_baggages'):
                continue
            args = function.get('arguments')
            if isinstance(args, str):
                args = json.loads(args)
            state = args if name == 'book_reservation' else reservations.get(args.get('reservation_id'), {})
            user = users.get(state.get('user_id'), {})
            cabin, passengers = state.get('cabin'), state.get('passengers')
            per_passenger = ALLOWANCE.get(user.get('membership'), {}).get(cabin)
            total, paid = args.get('total_baggages'), args.get('nonfree_baggages')
            check = {'event_id': f'm{index:03d}', 'tool': name, 'status': 'unknown'}
            if (per_passenger is not None and isinstance(passengers, list) and passengers
                    and type(total) is int and type(paid) is int):
                free = per_passenger * len(passengers)
                expected = max(0, total - free)
                check.update(status='satisfied' if paid == expected else 'violated',
                             membership=user['membership'], cabin=cabin,
                             passenger_count=len(passengers), free_total=free,
                             total_baggages=total, nonfree_baggages=paid, expected_nonfree=expected)
            checks.append(check)
    return checks
