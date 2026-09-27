import copy

import pytest

from tau3_grpo.analysis.simulator_drift import review_payload, scenario_lines, validate_review


def fixture():
    task = {'task': {'user_scenario': {'instructions': {'known_info': 'DOB: September 7, 1988',
            'task_instructions': 'Give that DOB when asked.\nAccept a later May return.'}}}}
    lines = scenario_lines(task)
    item = {'arm': 'secret_arm', 'row': {'reward': 0, 'trial': 2, 'simulation': {'messages': [
        {'role': 'assistant', 'content': 'What DOB?', 'reasoning': 'private'},
        {'role': 'user', 'content': 'September 7, 1988'},
        {'role': 'tool', 'content': 'future receipt', 'reward': 0}]}}}
    payload = review_payload(item, lines, {})
    packet = {'coverage': 'complete', 'findings': [{'kind': 'scenario_conflict', 'support': 'direct',
        'severity': 'minor', 'event_id': 'm001', 'user_quote': 'September 7, 1988',
        'scenario_refs': ['s000'], 'prior_event_refs': ['m000', 'm001'],
        'consequence': 'unknown', 'outcome_refs': [], 'reason': 'Scenario supplies DOB.'}],
        'uncertainties': [], 'summary': 'Scenario fact, not a new user invention.'}
    return payload, packet


def test_judge_input_blinding_and_original_facts_preserved():
    payload, packet = fixture()
    assert set(payload) == {'scenario_source_lines', 'contract_index', 'policy', 'events'}
    assert 'cannot modify the number of passengers' in payload['policy']
    assert payload['scenario_source_lines'][0]['text'] == 'DOB: September 7, 1988'
    assert 'reasoning' not in payload['events'][0] and 'reward' not in payload['events'][2]
    validate_review(packet, payload['events'], payload['scenario_source_lines'])


@pytest.mark.parametrize('change,match', [
    ({'event_id': 'm000'}, 'actual user'),
    ({'user_quote': 'fabricated'}, 'literal substring'),
    ({'scenario_refs': ['s999']}, 'scenario reference'),
    ({'prior_event_refs': ['m002']}, 'Future evidence'),
    ({'consequence': 'extra_write', 'outcome_refs': []}, 'without evidence'),
])
def test_evidence_role_quote_and_chronology(change, match):
    payload, packet = fixture()
    changed = copy.deepcopy(packet)
    changed['findings'][0].update(change)
    with pytest.raises(ValueError, match=match):
        validate_review(changed, payload['events'], payload['scenario_source_lines'])


def test_json_mode_is_opt_in_transport_setting():
    import asyncio
    import json
    import httpx
    from tau3_grpo.models.semantic_api import OpenAICompatibleSemanticModel
    from tau3_grpo.utils.hashing import sha256_json

    def handler(req):
        assert json.loads(req.content)['response_format'] == {'type': 'json_object'}
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                               'message': {'content': '{"ok":true}'}}]})
    client = OpenAICompatibleSemanticModel(base_url='https://api.deepseek.com', model='deepseek-flash',
        api_key='synthetic-test-only', response_format_json=True, transport=httpx.MockTransport(handler))
    request = {'system': 'JSON', 'visible_messages': [], 'prefix_sha256': sha256_json([])}
    assert asyncio.run(client.extract_json(request, user_payload={})) == {'ok': True}
