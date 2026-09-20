"""Opt-in question ledger: answering a field is not approval of an operation."""
from copy import deepcopy
import re

from tau3_grpo.algorithms.anchors.semantic_slots import fail
from tau3_grpo.utils.hashing import sha256_json


def normalize_question(data, event, messages):
    if set(data) != {'intent', 'proposal_id', 'focus'}:
        fail('invalid_focused_question_fields')
    intent, pid, focus = data['intent'], data['proposal_id'], data['focus']
    at = event.get('at')
    if type(at) is not int or not 0 <= at < len(messages):
        fail('invalid_question_position')
    text = messages[at].get('content')
    if (not isinstance(focus, str) or not focus.strip() or not isinstance(text, str)
            or focus not in text or not any(e.get('message_index') == at and focus in e.get('quote', '')
                                          for e in event.get('evidence', []))):
        fail('unsupported_question_focus')
    if intent == 'confirm_proposal':
        if not isinstance(pid, str) or not pid:
            fail('unbound_confirmation_question')
        if re.search(r'\b(?:which|choose|select|options?|user id|reservation id|booking id|passport|search)\b', focus, re.I):
            fail('confirmation_question_scope_conflict')
        if not re.search(r'\bconfirm\b|\bdo you approve\b|\bshall i proceed\b|\bwould you like (?:me )?to proceed\b', focus, re.I):
            fail('unsupported_confirmation_question')
    else:
        if pid is not None:
            fail('nonapproval_question_bound_to_proposal')
        patterns = {'choose_payment_method': r'\bpayment method\b',
                    'choose_option': r'\b(?:which|choose|select|prefer|options?)\b',
                    'explore_options': r'\b(?:explore|options?)\b',
                    'identity_user_id': r'\buser id\b',
                    'identity_reservation_id': r'\b(?:reservation|booking) id\b',
                    'identity_details': r'\b(?:user|reservation|booking) id\b',
                    'search_permission': r'\bsearch\b'}
        if intent not in patterns or not re.search(patterns[intent], focus, re.I):
            fail('unsupported_question_intent')
    return {'topic': intent, 'proposal_id': pid, 'focus': focus}


def check_question_coverage(messages, events):
    """A focused clause cannot hide another explicit question in that turn."""
    for at, message in enumerate(messages):
        if message.get('role') != 'assistant' or not isinstance(message.get('content'), str):
            continue
        rows = [e for e in events if e['at'] == at]
        if not any(e['kind'] in ('proposal', 'question') for e in rows):
            continue
        text = message['content']
        spans = []
        for e in rows:
            if e['kind'] == 'question':
                focus = e['data']['focus']
                spans.extend((m.start(), m.end()) for m in re.finditer(re.escape(focus), text))
        markers = list(re.finditer(r'\?|\bplease\s+(?:confirm|choose|select|provide|tell)\b', text, re.I))
        if any(not any(start <= m.start() < end for start, end in spans) for m in markers):
            fail('unaccounted_question_clause')


def check_consent_text(messages, event):
    data = event['data']
    text = messages[event['at']]['content']
    if data['status'] != 'approved':
        return
    if re.search(r"\b(?:not|no|never|don't|do not|unless|if|only)\b", text, re.I):
        fail('qualified_text_is_not_unconditional_approval')
    # A narrow execution directive, not permission to explain/search/check.
    execution_directive = re.search(
        r'\bgo ahead and (?:process|make|perform|execute) (?:the|this) '
        r'(?:change|cancellation|booking|upgrade|downgrade|removal)\b', text, re.I)
    if not (re.search(r'\b(?:yes|approve|confirm|proceed)\b', text, re.I) or execution_directive):
        fail('selection_is_not_approval')
    if data['binding'] == 'explicit' and not (
            (re.search(r'\b(?:approve|confirm|proceed)\b', text, re.I) or execution_directive)
            and re.search(r'\b(?:cancel|cancellation|upgrade|downgrade|change|booking|baggage|remove)\b', text, re.I)):
        fail('explicit_approval_requires_operation')


class QuestionLedger:
    def __init__(self):
        self.pending = {}
        self.answers = []
        self.context_seen = False
        self.reply_pending = {}
        self.answered_this_turn = set()
        self.questions_this_turn = set()

    def next_turn(self):
        self.finish_turn()
        self.reply_pending = deepcopy(self.pending)
        self.answered_this_turn.clear()
        self.questions_this_turn.clear()
        self.context_seen = False

    def finish_turn(self):
        if self.context_seen:
            # Exact background context still blocks approval references. Keep
            # explicit nonapproval questions asked alongside that background
            # so the next user may choose an option or supply a field.
            self.pending = {qid: q for qid, q in self.pending.items()
                            if qid in self.questions_this_turn and q['topic'] != 'confirm_proposal'}

    def clear(self):
        self.pending.clear()

    def remove_proposals(self, ids):
        self.pending = {k: q for k, q in self.pending.items() if q['proposal_id'] not in ids}

    def add(self, event):
        q = {**deepcopy(event['data']), 'id': event['id'], 'at': event['at']}
        # A new question about the same field/proposal replaces its old wording.
        self.pending = {k: old for k, old in self.pending.items()
                        if (old['topic'], old['proposal_id']) != (q['topic'], q['proposal_id'])}
        self.pending[q['id']] = q
        self.questions_this_turn.add(q['id'])

    def answer(self, event, messages):
        qid, value = event['data']['question_id'], event['data']['value']
        # Resolve against questions active at the start of this reply. A goal
        # replacement in the same user turn must not erase its answer target.
        # This snapshot never reactivates a proposal or grants consent.
        if qid not in self.reply_pending or qid in self.answered_this_turn:
            fail('answer_to_inactive_question')
        q = self.reply_pending[qid]
        if q['topic'] == 'confirm_proposal':
            fail('confirmation_requires_consent_event')
        if value != messages[event['at']]['content'] or not value.strip():
            fail('answer_must_preserve_current_text')
        if q['at'] != event['at'] - 1:
            fail('nonadjacent_question_answer')
        self.answers.append({'question': {k: q[k] for k in ('topic', 'focus')}, 'answer': value})
        self.answered_this_turn.add(qid)
        self.pending.pop(qid, None)

    def identity_answer(self, values):
        fields = {'identity_user_id': {'user_id_claim'},
                  'identity_reservation_id': {'reservation_id'},
                  'identity_details': {'user_id_claim', 'reservation_id'}}
        self.pending = {qid: q for qid, q in self.pending.items()
                        if q['topic'] not in fields or not fields[q['topic']] <= set(values)}

    def check_reply(self, pid, at):
        current = [q for q in self.reply_pending.values() if q['at'] == at - 1]
        if (self.context_seen or len(current) != 1 or current[0]['topic'] != 'confirm_proposal'
                or current[0]['proposal_id'] != pid or len(self.reply_pending) != 1):
            fail('ambiguous_compound_reply')

    def values(self, proposals):
        return sorted(({'topic': q['topic'], 'focus': q['focus'],
                        'proposal': proposals[q['proposal_id']]['value'] if q['proposal_id'] else None}
                       for q in self.pending.values()), key=sha256_json)
