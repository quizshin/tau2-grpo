import asyncio
import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

from tau3_grpo.algorithms.anchors.semantic_state import SemanticError, compile_state
from tau3_grpo.models.semantic_api import OpenAICompatibleSemanticModel
from tau3_grpo.models.semantic_extractor import build_request
from tau3_grpo.models.semantic_incremental import (
    IncrementalSemanticModel,
    append_delta,
    delta_request,
)
from tau3_grpo.utils.hashing import sha256_json


def fixture(name='approved'):
    cases = json.loads(Path('tests/fixtures/semantic_model_cases_20260914.json').read_text())['cases']
    case = next(c for c in cases if c['id'] == name)
    packets = json.loads(Path('tests/fixtures/semantic_model_responses_20260914.json').read_text())['packets']
    packet = next(p for p in packets if p['prefix_sha256'] == sha256_json(case['messages']))
    mapping = {e['id']: f"m{e['at']}:{i}" for i, e in enumerate(packet['events'])}
    def refs(value):
        if isinstance(value, str):
            return mapping.get(value, value)
        if isinstance(value, list):
            return [refs(v) for v in value]
        if isinstance(value, dict):
            return {k: refs(v) for k, v in value.items()}
        return value
    deltas = {}
    for at in range(len(case['messages'])):
        events = []
        for e in packet['events']:
            if e['at'] != at:
                continue
            events.append({'id': mapping[e['id']], 'kind': e['kind'],
                           'data': {} if e['kind'] == 'context' else refs(e['data']),
                           'evidence': list(dict.fromkeys(f"m{x['message_index']}" for x in e['evidence']))})
        key = sha256_json(case['messages'][:at+1])
        deltas[key] = {'schema': 'semantic_delta_v1', 'prefix_sha256': key, 'events': events}
    return case, packet, deltas


def client(deltas, **kwargs):
    def handler(req):
        payload = json.loads(req.content)
        user = json.loads(payload['messages'][1]['content'])
        assert 'visible_messages' not in user and 'reward' not in user
        assert user['current_at'] >= 0
        assert all(e['at'] < user['current_at'] for e in user['prior_events'])
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps(deltas[user['prefix_sha256']])}}]})
    api = OpenAICompatibleSemanticModel(base_url='https://example.com/v1', model='fixture',
                                        api_key='unit-test', transport=httpx.MockTransport(handler))
    return IncrementalSemanticModel(api, slot_schema=None, **kwargs)


@pytest.mark.parametrize('name', ['approved', 'changed_approved', 'later_condition', 'changed_pending'])
def test_append_reduction_matches_full_packet_including_consent_invalidation(name):
    case, packet, deltas = fixture(name)
    model = client(deltas)
    actual = asyncio.run(model.extract(build_request(case['messages'])))
    context = {k: case[k] for k in ('task_id', 'db_hash', 'policy_hash', 'remaining_turns')}
    assert compile_state(case['messages'], actual, **context)['key'] == compile_state(case['messages'], packet, **context)['key']


def test_stale_approval_still_rejected():
    case, packet, deltas = fixture('stale_approval')
    with pytest.raises(SemanticError, match='consent_to_inactive_proposal'):
        asyncio.run(client(deltas).extract(build_request(case['messages'])))


def test_concurrent_branches_share_only_identical_prefix_and_return_detached_logs():
    a, _, da = fixture('approved')
    b, _, db = fixture('amount')
    model = client(da | db)
    async def run():
        return await asyncio.gather(model.extract(build_request(a['messages'])),
                                    model.extract(build_request(b['messages'])))
    ra, rb = asyncio.run(run())
    assert model.attempted_calls == len(da | db)
    assert ra['events'][1]['data'] != rb['events'][1]['data']
    ra['events'][0]['data']['operation'] = 'other'
    assert rb['events'][0]['data']['operation'] == 'cancel'
    assert model.response_packets[ra['prefix_sha256']]['events'][0]['data']['operation'] == 'cancel'


def test_prefix_binding_no_history_rewrites_and_local_context():
    case, _, deltas = fixture()
    messages = case['messages'][:1]
    prior = {'schema': 'semantic_events_v1', 'prefix_sha256': sha256_json([]), 'events': []}
    request, payload = delta_request(messages, prior, slot_schema=None)
    delta = deltas[request['prefix_sha256']]
    bad = deepcopy(delta)
    bad['events'][0]['evidence'] = ['m99']
    with pytest.raises(SemanticError, match='reference'):
        append_delta(messages, prior, bad, payload['evidence_catalog'], slot_schema=None)
    bad = deepcopy(delta)
    bad['events'][0]['id'] = 'm99:rewrite'
    with pytest.raises(SemanticError, match='reference'):
        append_delta(messages, prior, bad, payload['evidence_catalog'], slot_schema=None)
    bad = deepcopy(delta)
    bad['events'][0].update(kind='context', data={'text': 'invented'})
    with pytest.raises(SemanticError, match='context_must_be_local'):
        append_delta(messages, prior, bad, payload['evidence_catalog'], slot_schema=None)
    assert prior['events'] == []


def test_request_budget_fail_closed():
    case, _, deltas = fixture()
    model = client(deltas, max_calls=1)
    with pytest.raises(SemanticError, match='budget_exhausted'):
        asyncio.run(model.extract(build_request(case['messages'])))
    assert model.attempted_calls == 1


def test_current_evidence_required_and_failed_append_is_atomic():
    case, _, deltas = fixture()
    messages = case['messages'][:2]
    model = client(deltas)
    prior = asyncio.run(model.extract(build_request(messages[:1])))
    original = deepcopy(prior)
    request, payload = delta_request(messages, prior, slot_schema=None)
    bad = deepcopy(deltas[request['prefix_sha256']])
    bad['events'][0]['evidence'] = ['m0']
    with pytest.raises(SemanticError, match='missing_current_evidence'):
        append_delta(messages, prior, bad, payload['evidence_catalog'], slot_schema=None)
    assert prior == original


def test_rejected_delta_is_recorded_and_descendants_do_not_retry():
    case, _, deltas = fixture('stale_approval')
    records = []
    model = client(deltas, on_delta=records.append)

    async def run():
        for messages in (case['messages'], case['messages'] + [{'role': 'user', 'content': 'Hello'}]):
            with pytest.raises(SemanticError, match='consent_to_inactive_proposal'):
                await model.extract(build_request(messages))

    asyncio.run(run())
    assert model.attempted_calls == len(deltas)
    assert records[-1]['error'] == 'consent_to_inactive_proposal'
    assert records[-1]['delta'] is not None
    assert sha256_json(case['messages']) not in model.response_packets
    records[-1]['delta']['events'].clear()
    assert model.delta_records[-1]['delta']['events']


def test_concurrent_branches_share_global_request_budget():
    a, _, da = fixture('approved')
    b, _, db = fixture('amount')
    model = client(da | db, max_calls=2)

    async def run():
        return await asyncio.gather(model.extract(build_request(a['messages'])),
                                    model.extract(build_request(b['messages'])),
                                    return_exceptions=True)

    results = asyncio.run(run())
    assert model.attempted_calls == 2
    assert all(isinstance(r, SemanticError) and 'budget_exhausted' in str(r) for r in results)


def test_incremental_audit_switch_persists_deltas_as_they_arrive(monkeypatch, tmp_path):
    from tau3_grpo.analysis.audit_semantic_api import run

    case, _, deltas = fixture('approved')
    case_path = tmp_path / 'cases.json'
    case_path.write_text(json.dumps({'cases': [case], 'pairs': []}))
    monkeypatch.setenv('TAU3_ENV_FILE', str(tmp_path / 'missing.env'))
    monkeypatch.setenv('TAU3_SEMANTIC_BASE_URL', 'https://example.com/v1')
    monkeypatch.setenv('TAU3_SEMANTIC_MODEL', 'fixture')
    monkeypatch.setenv('TAU3_SEMANTIC_API_KEY', 'unit-test')
    output = tmp_path / 'out'
    calls = 0

    def handler(req):
        nonlocal calls
        persisted = (output / 'deltas.jsonl').read_text().splitlines()
        assert len(persisted) == calls
        calls += 1
        payload = json.loads(json.loads(req.content)['messages'][1]['content'])
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
            'message': {'content': json.dumps(deltas[payload['prefix_sha256']])}}]})

    summary = asyncio.run(run({'enabled': True, 'provider': 'openai_compatible',
                               'cases': str(case_path), 'extraction_mode': 'incremental_v1'},
                              output, transport=httpx.MockTransport(handler)))
    assert summary['valid_cases'] == 1
    assert summary['api_request_attempts'] == len(deltas)
    assert summary['provenance']['history_rewrites'] is False
    assert not summary['training_enabled']
    assert len((output / 'deltas.jsonl').read_text().splitlines()) == len(deltas)
