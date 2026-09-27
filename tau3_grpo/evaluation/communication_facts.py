"""Arithmetic evidence from visible tool receipts, never hidden answers."""
from __future__ import annotations

import json
from datetime import datetime

VERSION = 'visible_airline_arithmetic_v1'
CURRENT_TIME = '2024-05-15T15:00:00'


def visible_facts(events):
    """Append audit facts; prior records always precede the cited write result.

    Prices are per passenger. A newly appended payment is a transaction delta,
    not the total reservation value. Facts do not assign a language-model score.
    """
    previous = {}
    facts = []
    calls = {}
    for event in events:
        if event.get('role') == 'assistant':
            for call in event.get('tool_calls') or []:
                calls[call.get('id')] = call.get('function', call)
        if event.get('role') != 'tool' or event.get('error') is not False:
            continue
        try:
            record = json.loads(event.get('content') or '')
        except (ValueError, TypeError):
            continue
        if not isinstance(record, dict) or not record.get('reservation_id'):
            continue
        rid = record['reservation_id']
        flights, passengers = record.get('flights'), record.get('passengers')
        if not isinstance(flights, list) or not isinstance(passengers, list):
            continue
        if not all(isinstance(f.get('price'), (int, float)) for f in flights):
            continue
        fact = dict(kind='reservation_snapshot', event_id=event['event_id'],
                    reservation_id=rid, call_id=event.get('tool_call_id'), cabin=record.get('cabin'),
                    passengers=len(passengers), booked_segment_prices=[f['price'] for f in flights],
                    booked_fare_total=sum(f['price'] for f in flights) * len(passengers),
                    payment_history=record.get('payment_history', []))
        if record.get('created_at'):
            try:
                created = datetime.fromisoformat(record['created_at'])
                age = (datetime.fromisoformat(CURRENT_TIME) - created).total_seconds()
                fact.update(created_at=record['created_at'], reference_time_est=CURRENT_TIME,
                            age_seconds=age, within_last_24_hours=0 <= age <= 86400)
            except (ValueError, TypeError):
                pass
        call = calls.get(event.get('tool_call_id'), {})
        if call.get('name') in ('update_reservation_flights', 'update_reservation_baggages', 'cancel_reservation') and rid in previous:
            old = previous[rid]
            payments = record.get('payment_history', [])
            old_payments = old.get('payment_history', [])
            if payments[:len(old_payments)] == old_payments:
                added = payments[len(old_payments):]
                old_total = sum(f['price'] for f in old['flights']) * len(old['passengers'])
                fact.update(kind='booking_write_transaction', before_event_id=old['_event_id'],
                            old_booked_fare_total=old_total,
                            appended_transactions=added,
                            signed_transaction_delta=sum(p['amount'] for p in added))
                if call.get('name') == 'update_reservation_flights':
                    fact['signed_fare_delta'] = fact['booked_fare_total'] - old_total
        if call.get('name') == 'book_reservation':
            fact.update(kind='new_booking_transaction',
                        appended_transactions=record.get('payment_history', []),
                        signed_transaction_delta=sum(p['amount'] for p in record.get('payment_history', [])))
        facts.append(fact)
        previous[rid] = dict(record, _event_id=event['event_id'])
    return dict(version=VERSION, current_time_est=CURRENT_TIME, facts=facts,
                interpretation='Positive appended amounts are additional charges; negative are refunds. '
                'Historical booked prices, not current market prices for the old cabin, determine old fare. '
                'These facts are evidence only; inspect pre-write disclosure and all assistant claims separately.')


def materialize_event_evidence(packet, events):
    """Resolve explicit IDs to verbatim content without fuzzy quote repair."""
    from tau3_grpo.evaluation.communication_contract import CommunicationUnresolved
    import copy
    packet = copy.deepcopy(packet)
    index = {e['event_id']: e for e in events}
    rows = packet.get('checks')
    scope = packet.get('simulator_scope')
    if not isinstance(rows, list) or not isinstance(scope, dict):
        raise CommunicationUnresolved('Missing checks or simulator scope')
    for row in rows + [scope]:
        refs = row.pop('evidence_event_ids', None)
        if not isinstance(refs, list) or 'evidence' in row:
            raise CommunicationUnresolved('Expected evidence_event_ids only')
        if any(not isinstance(ref, str) or ref not in index for ref in refs):
            raise CommunicationUnresolved('Invented evidence event')
        if len(set(refs)) != len(refs):
            raise CommunicationUnresolved('Duplicate evidence event')
        row['evidence'] = []
        for ref in refs:
            event = index[ref]
            text = event.get('content')
            if not isinstance(text, str) or not text.strip():
                text = json.dumps(event.get('tool_calls') or [], ensure_ascii=False)
            row['evidence'].append(dict(event_id=ref, quote=text))
    return packet
