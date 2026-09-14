"""Assumed-model tests: normalization, state transitions, scopes and evidence."""
from copy import deepcopy
import asyncio
import json
from pathlib import Path

import pytest
import yaml

from tau3_grpo.algorithms.anchors.semantic_state import SCHEMA, SemanticError, compile_state, validate_packet
from tau3_grpo.models.semantic_extractor import FixtureSemanticModel, build_request
from tau3_grpo.analysis.audit_semantic_model import run
from tau3_grpo.utils.hashing import sha256_json

CASES_PATH = Path('tests/fixtures/semantic_model_cases_20260914.json')
RESPONSES_PATH = Path('tests/fixtures/semantic_model_responses_20260914.json')


def sample(name):
    cases = json.loads(CASES_PATH.read_text())['cases']
    case = deepcopy(next(c for c in cases if c['id'] == name))
    packet = deepcopy(next(p for p in json.loads(RESPONSES_PATH.read_text())['packets']
                           if p['prefix_sha256'] == sha256_json(case['messages'])))
    return case, packet


def compile_case(case, packet):
    return compile_state(case['messages'], packet, task_id=case['task_id'], db_hash=case['db_hash'],
                         policy_hash=case['policy_hash'], remaining_turns=case['remaining_turns'])


def state(name):
    return compile_case(*sample(name))['state']


def statuses(name):
    return {row['operation_index']: row['status'] for consent in state(name)['consents'] for row in consent['operations']}


def test_full_mock_pipeline_matches_authored_contracts(tmp_path):
    config = yaml.safe_load(Path('configs/analysis/semantic_model_mock_20260914.yaml').read_text())
    report = asyncio.run(run(config, tmp_path / 'run'))
    assert report['passed_pairs'] == report['pairs'] == 34
    assert report['valid_cases'] == 48
    assert report['simulated_model'] and report['real_model_calls'] == 0 and not report['training_enabled']


def test_goal_replace_add_and_reaffirm_have_distinct_effects():
    assert len(state('replace_goal')['goals']) == 1
    assert len(state('add_goal')['goals']) == 2
    assert len(state('replace_goal')['history']) == 1
    assert compile_case(*sample('reaffirm'))['key'] == compile_case(*sample('approved'))['key']


def test_conditional_and_preference_are_not_conflated():
    assert statuses('conditional') == {0: 'conditional'}
    assert statuses('preference') == {0: 'approved'}
    assert state('later_condition')['consents'] == []


def test_partial_approval_accumulation_and_scoped_revocation():
    assert statuses('partial') == {0: 'approved'}
    assert statuses('both') == {0: 'approved', 1: 'approved'}
    assert statuses('accumulated') == statuses('both')
    assert statuses('revoke_bags') == {0: 'approved', 1: 'revoked'}
    assert compile_case(*sample('accumulated'))['key'] == compile_case(*sample('both'))['key']


def test_new_price_invalidates_and_old_commitment_remains():
    changed = state('changed_pending')
    assert not changed['consents']
    assert changed['proposals'][0]['operations'][0]['terms']['quoted_refund'] == '272'
    assert any(h.get('prior_proposal', {}).get('operations', [{}])[0].get('terms', {}).get('quoted_refund') == '100'
               for h in changed['history'])
    assert statuses('changed_approved') == {0: 'approved'}


@pytest.mark.parametrize('name,reason', [('ambiguous_yes', 'ambiguous_reply'), ('stale_approval', 'inactive_proposal'), ('unknown', 'unresolved_semantics')])
def test_ambiguous_scopes_abstain(name, reason):
    with pytest.raises(SemanticError, match=reason):
        compile_case(*sample(name))


def test_lookup_permission_never_becomes_write_approval():
    result = state('search_permission')
    assert result['goals'][0]['operation'] == 'lookup'
    assert not result['consents'] and not result['proposals']


@pytest.mark.parametrize('mutation', ['future', 'quote', 'offset', 'missing_text', 'unsafe_ignore', 'duplicate_id', 'unsupported_amount', 'unknown_field'])
def test_evidence_and_schema_fail_closed(mutation):
    case, packet = sample('approved')
    ev = packet['events'][0]['evidence'][0]
    if mutation == 'future': ev['message_index'] = len(case['messages'])
    elif mutation == 'quote': ev['quote'] = 'I never said this'
    elif mutation == 'offset': ev['start'] += 1
    elif mutation == 'missing_text': packet['events'].pop()
    elif mutation == 'unsafe_ignore': packet['events'][0].update(kind='ignore', data={})
    elif mutation == 'duplicate_id': packet['events'][1]['id'] = packet['events'][0]['id']
    elif mutation == 'unsupported_amount': packet['events'][1]['data']['operations'][0]['terms']['quoted_refund'] = '999'
    elif mutation == 'unknown_field': packet['reward'] = 1
    with pytest.raises(SemanticError): compile_case(case, packet)


def test_local_reference_ids_and_decimal_format_do_not_fragment_state():
    case, packet = sample('approved'); expected = compile_case(case, packet)['key']
    packet['events'][0]['id'] = 'goal_with_new_id'
    packet['events'][1]['id'] = 'offer_with_new_id'
    packet['events'][1]['data']['goal_ids'] = ['goal_with_new_id']
    packet['events'][1]['data']['operations'][0]['goal_ids'] = ['goal_with_new_id']
    packet['events'][2]['data']['proposal_id'] = 'offer_with_new_id'
    packet['events'][1]['data']['operations'][0]['terms']['quoted_refund'] = '100.00'
    packet['events'][1]['data']['operations'][0]['terms']['currency'] = 'usd'
    assert compile_case(case, packet)['key'] == expected


@pytest.mark.parametrize('field', ['task_id', 'db_hash', 'policy_hash', 'remaining_turns'])
def test_environment_and_budget_are_part_of_candidate_key(field):
    case, packet = sample('approved'); expected = compile_case(case, packet)['key']
    case[field] = 1 if field == 'remaining_turns' else 'different'
    assert compile_case(case, packet)['key'] != expected


def test_model_request_never_contains_expected_relation_or_reward():
    case, _ = sample('approved'); raw = deepcopy(case['messages'])
    raw[0]['reward'] = 1; raw[0]['expected_relation'] = 'merge'; raw[0]['hidden_goal'] = 'secret'
    request = build_request(raw)
    assert set(request) == {'system', 'schema', 'prefix_sha256', 'visible_messages'}
    assert request['visible_messages'] == case['messages']


def test_missing_model_fixture_fails_instead_of_fabricating_empty_state():
    model = FixtureSemanticModel(RESPONSES_PATH)
    with pytest.raises(LookupError):
        asyncio.run(model.extract(build_request([{'role': 'user', 'content': 'Unseen request'}])))


def test_only_simulated_provider_can_be_run(tmp_path):
    with pytest.raises(ValueError, match='Only simulated'):
        asyncio.run(run({'enabled': True, 'provider': 'real'}, tmp_path / 'run'))
    assert not (tmp_path / 'run').exists()


def test_condition_applies_only_to_affected_operation():
    case, packet = sample('both')
    text = 'Approve both changes, but add the bags only if there is no bag fee.'
    case['messages'][2]['content'] = text
    packet['prefix_sha256'] = sha256_json(case['messages'])
    evidence = [{'message_index': 2, 'start': 0, 'end': len(text), 'quote': text}]
    packet['events'][-1]['evidence'] = evidence
    condition = {'id': 'cb', 'kind': 'constraint', 'at': 2, 'data': {'goal_ids': ['gb'],
                 'predicate': {'baggage_fee': '0'}, 'strength': 'requirement', 'status': 'active', 'supersedes': []},
                 'evidence': evidence}
    packet['events'].insert(-1, condition)
    output = compile_case(case, packet)['state']['consents'][0]['operations']
    assert [r['status'] for r in output] == ['approved', 'conditional']


def test_constraint_withdrawal_requires_prior_reference_and_reapproval():
    case, packet = sample('conditional')
    text = 'I withdraw the no-fee condition.'
    case['messages'].append({'role': 'user', 'content': text})
    packet['events'].append({'id': 'cw', 'kind': 'constraint', 'at': 3,
      'data': {'goal_ids': ['g'], 'predicate': {'fee': {'op': 'eq', 'amount': '0', 'currency': 'USD'}},
               'strength': 'requirement', 'status': 'withdrawn', 'supersedes': ['c']},
      'evidence': [{'message_index': 3, 'start': 0, 'end': len(text), 'quote': text}]})
    packet['prefix_sha256'] = sha256_json(case['messages'])
    result = compile_case(case, packet)['state']
    assert not result['consents'] and result['constraints'][0]['status'] == 'withdrawn'
    packet['events'][-1]['data']['supersedes'] = []
    with pytest.raises(SemanticError, match='unbound_constraint_resolution'): compile_case(case, packet)


def test_observed_tool_leaf_can_support_goal_but_not_invent_user_authorization():
    messages = [
        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'r', 'name': 'get_reservation_details', 'arguments': {'reservation_id': 'ABC123'}}]},
        {'role': 'tool', 'id': 'r', 'content': '{"reservation_id":"ABC123","user_id":"alice_123"}'},
        {'role': 'user', 'content': 'Cancel that reservation.'}]
    packet = {'schema': SCHEMA, 'prefix_sha256': sha256_json(messages), 'events': [
        {'id': 'g', 'kind': 'goal', 'at': 2, 'data': {'operation': 'cancel', 'target': {'reservation_id': 'ABC123'}, 'relation': 'add', 'replaces': []},
         'evidence': [{'message_index': 2, 'start': 0, 'end': len(messages[2]['content']), 'quote': messages[2]['content']},
                      {'message_index': 1, 'pointer': '/reservation_id', 'value': 'ABC123'}]}]}
    assert validate_packet(messages, packet)
    packet['events'][0]['evidence'][1]['value'] = 'XYZ999'
    with pytest.raises(SemanticError, match='unsupported_tool_evidence'): validate_packet(messages, packet)


def test_goal_replacement_keeps_unapproved_financial_promise():
    case, packet = sample('approved')
    case['messages'] = case['messages'][:2] + [{'role': 'user', 'content': 'Instead, change this reservation to economy.'}]
    packet['events'] = packet['events'][:2] + [{'id': 'g2', 'kind': 'goal', 'at': 2,
        'data': {'operation': 'change_cabin', 'target': {'reservation_id': 'ABC123', 'target_cabin': 'economy'},
                 'relation': 'replace', 'replaces': ['g']},
        'evidence': [{'message_index': 2, 'start': 0, 'end': len(case['messages'][2]['content']), 'quote': case['messages'][2]['content']}]}]
    packet['prefix_sha256'] = sha256_json(case['messages'])
    result = compile_case(case, packet)['state']
    assert not result['proposals']
    assert any(h.get('prior_proposal', {}).get('operations', [{}])[0].get('terms', {}).get('quoted_refund') == '100'
               for h in result['history'])


def test_compiled_state_reaches_real_gigpo_step_advantage_without_gpu():
    import numpy as np
    from tau3_grpo.algorithms.anchors.semantic_state import scope_candidate_key
    from tau3_grpo.algorithms.tau_gigpo import compute_tau_gigpo_advantage, steps_from_anchor_payload

    def advantage(names, rewards, groups):
        keys = [scope_candidate_key(compile_case(*sample(name))['key'], group) for name, group in zip(names, groups)]
        steps = steps_from_anchor_payload([[k] for k in keys], [[(0, 1)], [(0, 1)]])
        return compute_tau_gigpo_advantage(rewards, groups, steps, response_length=1)

    # These are synthetic 1/0 outcomes, never an empirical RL improvement claim.
    adv, stats = advantage(['approved', 'paraphrased'], [1, 0], ['g', 'g'])
    np.testing.assert_allclose(adv, [[1.], [-1.]])  # episode ±.5 plus step ±.5
    assert stats.usable_anchor_groups == 1
    adv, stats = advantage(['approved', 'amount'], [1, 0], ['g', 'g'])
    np.testing.assert_allclose(adv, [[.5], [-.5]])  # different quote: episode only
    assert stats.usable_anchor_groups == 0
    adv, _ = advantage(['approved', 'paraphrased'], [1, 1], ['g', 'g'])
    np.testing.assert_allclose(adv, [[0.], [0.]])
    adv, stats = advantage(['approved', 'paraphrased'], [1, 0], ['g1', 'g2'])
    np.testing.assert_allclose(adv, [[0.], [0.]])
    assert stats.usable_anchor_groups == 0


def test_one_reply_can_approve_one_operation_and_reject_another():
    case, packet = sample('both')
    text = 'Yes to the date change, no to the bags.'
    case['messages'][2]['content'] = text
    evidence = [{'message_index': 2, 'start': 0, 'end': len(text), 'quote': text}]
    approved = packet['events'][-1]
    approved['evidence'] = evidence; approved['data']['operation_indices'] = [0]
    rejected = deepcopy(approved); rejected['id'] = 'reject_bags'
    rejected['data'].update(operation_indices=[1], status='rejected')
    packet['events'].append(rejected); packet['prefix_sha256'] = sha256_json(case['messages'])
    rows = compile_case(case, packet)['state']['consents'][0]['operations']
    assert [row['status'] for row in rows] == ['approved', 'rejected']


def test_reaffirm_and_approve_in_same_reply_keeps_reference():
    case, packet = sample('approved')
    text = 'Yes, I still want to cancel ABC123; please proceed with your offer.'
    case['messages'][2]['content'] = text
    evidence = [{'message_index': 2, 'start': 0, 'end': len(text), 'quote': text}]
    packet['events'][-1]['evidence'] = evidence
    reaffirm = deepcopy(packet['events'][0]); reaffirm.update(id='g2', at=2, evidence=evidence)
    reaffirm['data'].update(relation='reaffirm', replaces=['g'])
    packet['events'].insert(-1, reaffirm); packet['prefix_sha256'] = sha256_json(case['messages'])
    assert compile_case(case, packet)['key'] == compile_case(*sample('approved'))['key']
