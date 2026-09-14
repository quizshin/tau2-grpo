"""Conservative, whole-utterance Airline semantic normalization.

A rule must consume the entire utterance. Unparsed clauses remain opaque; slots
alone must never erase negation, conditions, target changes or payment details.
This is a bounded grammar, not a general natural-language authorization model.
"""
from __future__ import annotations

from decimal import Decimal
import re
from typing import Any

_ID = r'[A-Za-z0-9_\-]+'


def _match(pattern: str, text: str):
    return re.fullmatch(pattern, text, flags=re.IGNORECASE)


def _known(kind: str, **slots: Any):
    return {'kind': kind, 'slots': slots}


def normalize_utterance(role: str, text: str) -> dict:
    text = ' '.join(text.replace('’', "'").split()).strip()
    # Terminal punctuation is syntax only for these fully matched grammars.
    sentence = text.rstrip('.!?').strip()
    if role == 'assistant' and _match(
        r'(?:(?:hi|hello|hey)[!, .]*)?(?:how (?:can|may) i (?:help|assist) you(?: today)?|what can i (?:help|assist) you with(?: today)?)', sentence
    ):
        return _known('open_help_question')
    if role == 'user':
        # A questioning 'Yes?' or uncertain identity is not affirmative consent.
        if '?' in text and not _match(r'(?:can|could) you cancel [^?]+\?', text):
            return {'kind':'opaque','text':text}
        if _match(r'(?:yes|yes,? please|yes,? (?:please )?(?:go ahead|proceed)|(?:please )?(?:go ahead|proceed)|i (?:confirm|approve))', sentence):
            return _known('assent_to_current_proposal')
        if _match(r'(?:no|no,? (?:please )?(?:do not|don\'t) (?:proceed|go ahead)|(?:please )?(?:do not|don\'t) (?:proceed|go ahead)|i (?:do not|don\'t) (?:confirm|approve))', sentence):
            return _known('deny_current_proposal')
        if _match(r'(?:wait|please wait|hold on|hold on a moment|please hold on)', sentence):
            return _known('pause')
        if _match(r'(?:i (?:withdraw|revoke) my (?:consent|approval)|(?:wait,? )?i (?:withdraw|revoke) my (?:consent|approval))', sentence):
            return _known('revoke_prior_consent')
        identity = _match(rf'(?:my )?user (?:id|identifier) (?:is|:) (?P<user>{_ID})(?:[,.]? (?:and )?(?:my )?(?:reservation|booking) (?:id|number|code) (?:is|:) (?P<reservation>{_ID}))?', sentence)
        if identity:
            return _known('identity', **identity.groupdict())
        reservation = _match(rf'(?:my )?(?:reservation|booking) (?:id|number|code) (?:is|:) (?P<reservation>{_ID})', sentence)
        if reservation:
            return _known('reservation_reference', **reservation.groupdict())
        request = _match(rf'(?:(?:i would like to|i\'d like to|i want to|could you|can you|please) )?cancel (?:my |the )?(?:reservation|booking)(?: (?P<reservation>{_ID}))?(?: (?:because of|due to) (?P<reason>a change of plans|change of plans|illness))?', sentence)
        if request:
            slots=request.groupdict()
            if slots['reason'] is not None:
                slots['reason']=slots['reason'].lower()
                if slots['reason']=='a change of plans':slots['reason']='change of plans'
            return _known('request_cancellation', **slots)
        # Explicit read permission is NOT generic permission to modify a booking.
        if _match(r'yes,? (?:please )?(?:go ahead and )?(?:search for|check|look up) the economy price for (?:that|the) flight(?: so we can see the refund amount)?', sentence):
            return _known('permit_price_lookup', cabin='economy', target='context_flight')
    if role == 'assistant':
        ask_user = _match(r'(?:could you|can you|please) (?:please )?(?:provide|share|tell me) (?:your )?user (?:id|identifier)', sentence)
        if ask_user:
            return _known('request_user_id')
        ask_reservation = _match(r'(?:could you|can you|please) (?:please )?(?:provide|share|tell me) (?:your )?(?:reservation|booking) (?:id|number|code)', sentence)
        if ask_reservation:
            return _known('request_reservation_id')
        # Complete proposals: require explicit target, refund amount/currency and
        # payment destination. No filling missing fields from hidden task data.
        patterns = [
            rf'cancel (?:reservation|booking) (?P<reservation>{_ID}) (?:for|with) (?:a )?(?P<currency>\$|USD|EUR|€|GBP|£)\s*(?P<amount>\d+(?:\.\d+)?) refund to (?:card|payment method) (?P<payment>{_ID})[.;]? (?:do you confirm|please confirm)',
            rf'please confirm cancellation of (?:reservation|booking) (?P<reservation>{_ID})[;,:]? refund (?P<currency>\$|USD|EUR|€|GBP|£)\s*(?P<amount>\d+(?:\.\d+)?) to (?:card|payment method) (?P<payment>{_ID})',
        ]
        for pattern in patterns:
            proposal=_match(pattern,sentence)
            if proposal:
                slots=proposal.groupdict();slots['currency']={'$':'USD','€':'EUR','£':'GBP'}.get(slots['currency'],slots['currency'].upper())
                slots['amount']=format(Decimal(slots['amount']).normalize(),'f')
                return _known('propose_cancellation',**slots)
    return {'kind':'opaque','text':text}
