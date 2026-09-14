"""Evidence-checked semantic events and deterministic decision-state reduction.

An LLM supplies semantic interpretations. This module verifies their visible
sources and enforces reference/scope/consent transitions; source validity alone
is not proof that the interpretation is correct. No LLM is called by the hook.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import math
import re

from tau3_grpo.algorithms.anchors.semantic import normalize_utterance
from tau3_grpo.algorithms.anchors.evidence import decision_evidence
from tau3_grpo.utils.hashing import sha256_json

SCHEMA = 'semantic_events_v1'
OPERATIONS = {'cancel', 'change_flights', 'change_cabin', 'change_baggage', 'book', 'lookup', 'other'}
FIELDS = {
    'goal': {'operation', 'target', 'relation', 'replaces'},
    'constraint': {'goal_ids', 'predicate', 'strength', 'status', 'supersedes'},
    'proposal': {'goal_ids', 'operations', 'supersedes'},
    'consent': {'proposal_id', 'operation_indices', 'status', 'binding'},
    'question': {'topic', 'proposal_id'},
    'context': {'text'},
    'unknown': {'reason'},
    'ignore': set(),
}


class SemanticError(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise SemanticError(reason)


def _exact_fields(value, fields, name):
    _require(isinstance(value, dict) and set(value) == fields, 'schema:' + name)


def _strings(value):
    return isinstance(value, list) and all(isinstance(x, str) and x for x in value) and len(value) == len(set(value))


def _json_tree(value):
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_tree(v) for k, v in value.items())
    if isinstance(value, list):
        return all(_json_tree(v) for v in value)
    return value is None or type(value) in (str, bool, int) or type(value) is float and math.isfinite(value)


def canonical_semantics(value, field=''):
    if isinstance(value, dict):
        return {key: canonical_semantics(child, key) for key, child in value.items()}
    if isinstance(value, list):
        return [canonical_semantics(child) for child in value]
    if field == 'currency' and isinstance(value, str):
        return value.upper()
    if field in ('quoted_refund', 'quoted_charge', 'amount', 'charge', 'refund', 'fee'):
        if type(value) in (int, float, str) and re.fullmatch(r'-?\d+(?:\.\d+)?', str(value)):
            number = Decimal(str(value))
            return format(number.normalize(), 'f')
    return value


def validate_packet(messages, packet):
    """Strict schema, chronological citations, and complete text accounting."""
    _exact_fields(packet, {'schema', 'prefix_sha256', 'events'}, 'packet')
    _require(packet['schema'] == SCHEMA and packet['prefix_sha256'] == sha256_json(messages), 'prefix_mismatch')
    _require(isinstance(packet['events'], list), 'events_not_list')
    ids, covered, previous = set(), {}, -1
    observed_facts = None
    for event in packet['events']:
        _exact_fields(event, {'id', 'kind', 'at', 'data', 'evidence'}, 'event')
        kind, at, data = event['kind'], event['at'], event['data']
        _require(isinstance(event['id'], str) and event['id'] and event['id'] not in ids, 'duplicate_event_id')
        ids.add(event['id'])
        _require(kind in FIELDS and type(at) is int and 0 <= at < len(messages) and at >= previous, 'event_order')
        previous = at
        _exact_fields(data, FIELDS[kind], kind)
        _require(_json_tree(data), 'non_json_data')
        _require(isinstance(event['evidence'], list) and event['evidence'], 'missing_evidence')
        current_text_evidence = []
        for evidence in event['evidence']:
            if isinstance(evidence, dict) and 'pointer' in evidence:
                _exact_fields(evidence, {'message_index', 'pointer', 'value'}, 'tool_evidence')
                index = evidence['message_index']
                _require(type(index) is int and 0 <= index <= at, 'future_tool_evidence')
                if observed_facts is None:
                    from tau3_grpo.algorithms.anchors.grounded import extract
                    observed_facts = extract(messages)['facts']
                _require(any(f['evidence']['message_index'] == index
                             and f['evidence']['pointer'] == evidence['pointer']
                             and sha256_json(f['value']) == sha256_json(evidence['value'])
                             for f in observed_facts), 'unsupported_tool_evidence')
                continue
            _exact_fields(evidence, {'message_index', 'start', 'end', 'quote'}, 'evidence')
            index, start, end = evidence['message_index'], evidence['start'], evidence['end']
            _require(type(index) is int and 0 <= index <= at, 'future_evidence')
            text = messages[index].get('content')
            _require(messages[index].get('role') in ('user', 'assistant') and isinstance(text, str), 'evidence_role')
            _require(type(start) is int and type(end) is int and 0 <= start < end <= len(text)
                     and text[start:end] == evidence['quote'], 'evidence_span')
            # Earlier supporting quotes may be reused; only the event's own
            # message contributes to that message's completeness accounting.
            if index == at:
                covered.setdefault(at, set()).update(range(start, end))
                current_text_evidence.append(evidence['quote'])
        _require(current_text_evidence, 'missing_current_evidence')
        role = messages[at]['role']
        if kind in ('goal', 'constraint', 'consent'):
            _require(role == 'user', 'event_requires_user')
        if kind in ('proposal', 'question'):
            _require(role == 'assistant', 'event_requires_assistant')
        if kind == 'goal':
            _require(data['operation'] in OPERATIONS and isinstance(data['target'], dict)
                     and data['target'] and _strings(data['replaces'])
                     and data['relation'] in ('add', 'replace', 'reaffirm'), 'invalid_goal')
            _require((data['relation'] == 'add' and not data['replaces'])
                     or (data['relation'] == 'replace' and bool(data['replaces']))
                     or (data['relation'] == 'reaffirm' and len(data['replaces']) == 1), 'invalid_goal_relation')
        elif kind == 'constraint':
            _require(_strings(data['goal_ids']) and data['goal_ids'] and isinstance(data['predicate'], dict)
                     and data['predicate'] and data['status'] in ('active', 'withdrawn', 'satisfied')
                     and data['strength'] in ('requirement', 'preference')
                     and _strings(data['supersedes']), 'invalid_constraint')
        elif kind == 'proposal':
            _require(_strings(data['goal_ids']) and _strings(data['supersedes'])
                     and isinstance(data['operations'], list) and data['operations'], 'invalid_proposal')
            for operation in data['operations']:
                _exact_fields(operation, {'operation', 'target', 'terms', 'goal_ids'}, 'proposal_operation')
                _require(operation['operation'] in OPERATIONS and isinstance(operation['target'], dict)
                         and operation['target'] and isinstance(operation['terms'], dict)
                         and _strings(operation['goal_ids'])
                         and set(operation['goal_ids']) <= set(data['goal_ids']), 'invalid_operation')
                # Quoted monetary amounts must occur in the cited assistant
                # text, never be substituted with a tool-calculated amount.
                from tau3_grpo.algorithms.anchors.grounded import text_claims
                supported_money = []
                for ev in event['evidence']:
                    if 'start' not in ev:
                        continue
                    source = messages[ev['message_index']]
                    if source['role'] != 'assistant':
                        continue
                    supported_money.extend(c['value'] for c in text_claims(source, ev['message_index'])
                                           if c['kind'] == 'money_mention'
                                           and ev['start'] <= c['evidence']['start'] < c['evidence']['end'] <= ev['end'])
                terms = canonical_semantics(operation['terms'])
                for field in ('quoted_refund', 'quoted_charge'):
                    if field in terms:
                        _require({'currency': terms.get('currency'), 'amount': terms[field]} in supported_money,
                                 'unsupported_communicated_amount')
        elif kind == 'consent':
            indices = data['operation_indices']
            _require(isinstance(data['proposal_id'], str) and isinstance(indices, list) and indices
                     and all(type(x) is int and x >= 0 for x in indices) and len(indices) == len(set(indices))
                     and data['status'] in ('approved', 'rejected', 'paused', 'revoked', 'conditional')
                     and data['binding'] in ('reply', 'explicit'), 'invalid_consent')
        elif kind == 'question':
            _require(isinstance(data['topic'], str) and data['topic']
                     and (data['proposal_id'] is None or isinstance(data['proposal_id'], str)), 'invalid_question')
        elif kind == 'context':
            # No free semantic summary may silently discard a qualification.
            _require(data['text'] == messages[at]['content'], 'context_must_be_verbatim')
        elif kind == 'unknown':
            _require(isinstance(data['reason'], str) and data['reason'], 'unknown_reason')
        elif kind == 'ignore':
            # Only a full, known greeting can be ignored mechanically.
            _require(role == 'assistant' and normalize_utterance(role, messages[at]['content'])['kind'] == 'open_help_question', 'unsafe_ignore')
    for index, message in enumerate(messages):
        if message.get('role') in ('assistant', 'user') and isinstance(message.get('content'), str):
            required = {i for i, char in enumerate(message['content']) if not char.isspace()}
            _require(required <= covered.get(index, set()), f'unaccounted_text:{index}')
    return True


def compile_state(messages, packet, *, task_id, db_hash, policy_hash, remaining_turns=None):
    """Construct an offline candidate key; no claim of model semantic accuracy.

    IDs and evidence positions are local references and disappear from the key.
    Exact unparsed context and historical financial proposals remain protected.
    Unknown events or invalid references yield a hard validation failure.
    """
    validate_packet(messages, packet)
    _require(bool(task_id) and bool(db_hash) and bool(policy_hash), 'missing_environment_context')
    _require(remaining_turns is None or type(remaining_turns) is int and remaining_turns >= 0, 'invalid_remaining_turns')
    goals, constraints, proposals, consents = {}, {}, {}, {}
    active_goals, active_proposals, history, opaque = set(), set(), [], []
    open_question = None
    turn_at, turn_reference = None, None

    def refs(values, pool):
        _require(all(value in pool for value in values), 'unknown_reference')

    def goal_values(values):
        return sorted((goals[value]['value'] for value in values), key=sha256_json)

    def goal_refs(values):
        refs(values, goals)
        return sorted({goals[value]['root'] for value in values})

    def invalidate(proposal_ids, reason):
        for pid in proposal_ids:
            if pid in consents:
                history.append({'prior_consent': consents.pop(pid), 'invalidation': reason})

    for event in packet['events']:
        kind, data, eid, at = event['kind'], canonical_semantics(deepcopy(event['data'])), event['id'], event['at']
        if at != turn_at:
            turn_at, turn_reference = at, deepcopy(open_question)
        if kind == 'unknown':
            raise SemanticError('unresolved_semantics:' + eid)
        if kind == 'goal':
            data['replaces'] = goal_refs(data['replaces'])
            _require(set(data['replaces']) <= active_goals, 'replacing_inactive_goal')
            value = {'operation': data['operation'], 'target': data['target']}
            if data['relation'] == 'reaffirm':
                old = data['replaces'][0]
                _require(sha256_json(goals[old]['value']) == sha256_json(value), 'reaffirm_changed_goal')
                goals[eid] = goals[old]
                continue
            affected = {pid for pid in active_proposals if set(proposals[pid]['goals']) & set(data['replaces'])}
            invalidate(affected, 'goal_replaced')
            for pid in sorted(affected, key=lambda pid: sha256_json(proposals[pid]['value'])):
                history.append({'prior_proposal': proposals[pid]['value']})
            active_proposals -= affected
            for old in data['replaces']:
                history.append({'replaced_goal': goals[old]['value']})
            active_goals -= set(data['replaces'])
            goals[eid] = {'value': value, 'at': at, 'root': eid}
            active_goals.add(eid); open_question = None; turn_reference = None
        elif kind == 'constraint':
            data['goal_ids'] = goal_refs(data['goal_ids']); refs(data['supersedes'], constraints)
            _require(set(data['goal_ids']) <= active_goals, 'constraint_on_inactive_goal')
            _require(data['status'] == 'active' or data['supersedes'], 'unbound_constraint_resolution')
            for old in data['supersedes']:
                _require(set(constraints[old]['goals']) == set(data['goal_ids']), 'constraint_resolution_scope_changed')
                if data['status'] != 'active':
                    old_value = constraints[old]['value']
                    _require(sha256_json(old_value['predicate']) == sha256_json(data['predicate'])
                             and old_value['strength'] == data['strength'], 'constraint_resolution_predicate_changed')
                history.append({'prior_constraint': constraints.pop(old)['value']})
            value = {'goals': goal_values(data['goal_ids']), 'predicate': data['predicate'],
                     'strength': data['strength'], 'status': data['status']}
            constraints[eid] = {'value': value, 'goals': data['goal_ids']}
            affected = {pid for pid in active_proposals if set(proposals[pid]['goals']) & set(data['goal_ids'])}
            invalidate(affected, 'constraint_changed')
            # A condition and a conditional yes in the SAME user turn may
            # still refer to the immediately preceding offer. A later bare yes
            # fails the adjacency check below.
        elif kind == 'proposal':
            data['goal_ids'] = goal_refs(data['goal_ids']); refs(data['supersedes'], proposals)
            _require(set(data['goal_ids']) <= active_goals and set(data['supersedes']) <= active_proposals, 'inactive_proposal_scope')
            overlapping = {pid for pid in active_proposals if set(proposals[pid]['goals']) & set(data['goal_ids'])}
            offered_scopes = {sha256_json([op['operation'], op['target']]) for op in data['operations']}
            overlapping |= {pid for pid in active_proposals if offered_scopes &
                            {sha256_json([op['operation'], op['target']]) for op in proposals[pid]['value']['operations']}}
            _require(overlapping <= set(data['supersedes']), 'undeclared_proposal_replacement')
            invalidate(data['supersedes'], 'proposal_changed')
            for old in data['supersedes']:
                history.append({'prior_proposal': proposals[old]['value']})
            active_proposals -= set(data['supersedes'])
            operation_goals = [goal_refs(op['goal_ids']) for op in data['operations']]
            operations = [{k: v for k, v in op.items() if k != 'goal_ids'}
                          | {'goals': goal_values(ids)} for op, ids in zip(data['operations'], operation_goals)]
            value = {'goals': goal_values(data['goal_ids']), 'operations': operations}
            proposals[eid] = {'value': value, 'goals': data['goal_ids'], 'at': at, 'operation_goals': operation_goals}
            active_proposals.add(eid); open_question = {'proposal_id': eid, 'at': at, 'topic': 'confirm_proposal'}
        elif kind == 'question':
            pid = data['proposal_id']
            if pid is not None:
                _require(pid in active_proposals, 'question_about_inactive_proposal')
            open_question = {'proposal_id': pid, 'at': at, 'topic': data['topic']}
        elif kind == 'consent':
            pid = data['proposal_id']
            _require(pid in active_proposals, 'consent_to_inactive_proposal')
            proposal = proposals[pid]
            _require(all(i < len(proposal['value']['operations']) for i in data['operation_indices']), 'consent_scope_out_of_range')
            if data['binding'] == 'reply':
                reference = open_question or turn_reference
                _require(reference is not None and reference['proposal_id'] == pid, 'ambiguous_reply')
                # No intervening question, tool, or extra conversational turn
                # may change the referent of a bare yes.
                _require(reference['at'] == at - 1, 'nonadjacent_reply')
            prior = consents.get(pid, {'proposal': proposal['value'], 'operations': []})
            operation_consents = {row['operation_index']: row for row in prior['operations']}
            for index in data['operation_indices']:
                active_conditions = [c['value'] for c in constraints.values()
                                     if c['value']['status'] == 'active' and c['value']['strength'] == 'requirement'
                                     and set(c['goals']) & set(proposal['operation_goals'][index])]
                status = 'conditional' if data['status'] == 'approved' and active_conditions else data['status']
                _require(status != 'conditional' or active_conditions, 'unexplained_condition')
                operation_consents[index] = {'operation_index': index, 'status': status,
                                             'conditions': sorted(active_conditions, key=sha256_json)}
            consents[pid] = {'proposal': proposal['value'],
                             'operations': [operation_consents[i] for i in sorted(operation_consents)]}
            open_question = None
        elif kind == 'context':
            opaque.append({'role': messages[at]['role'], 'text': data['text']})
            if messages[at]['role'] == 'user':
                invalidate(list(consents), 'uninterpreted_user_context')
            # Conservatively invalidate the referent after any uninterpreted turn.
            open_question = None
            turn_reference = None
    knowledge = decision_evidence(messages, version='v2')
    state = {'goals': goal_values(active_goals),
             'constraints': sorted((c['value'] for c in constraints.values()), key=sha256_json),
             'proposals': sorted((proposals[pid]['value'] for pid in active_proposals), key=sha256_json),
             'consents': sorted(consents.values(), key=sha256_json),
             'history': history, 'opaque_context': opaque,
             'pending_question': ({'topic': open_question['topic'],
                                   'proposal': proposals[open_question['proposal_id']]['value']
                                   if open_question['proposal_id'] else None} if open_question else None),
             'read_hash': knowledge.read_hash, 'tool_event_hash': knowledge.tool_event_hash,
             'remaining_turns': remaining_turns}
    key = 'semantic:v1:' + sha256_json({'task': task_id, 'db': db_hash, 'policy': policy_hash, 'state': state})
    return {'key': key, 'state': state, 'evidence_valid': True,
            'semantic_accuracy_verified': False, 'training_enabled': False}


def scope_candidate_key(key, episode_group_id):
    """Namespace a candidate by the sampled group before any GiGPO comparison."""
    _require(isinstance(key, str) and key.startswith('semantic:v1:'), 'invalid_candidate_key')
    _require(isinstance(episode_group_id, str) and bool(episode_group_id), 'missing_episode_group')
    return 'semantic:group:v1:' + sha256_json([episode_group_id, key])
