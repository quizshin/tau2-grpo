"""Mock HTTP only: no API key, external inference or GPU required."""
import asyncio
import json
from pathlib import Path

import httpx
import pytest
import yaml

from tau3_grpo.algorithms.anchors.semantic_state import SemanticError
from tau3_grpo.analysis.audit_semantic_api import run
from tau3_grpo.models.semantic_api import OpenAICompatibleSemanticModel, SemanticAPIError
from tau3_grpo.models.semantic_extractor import build_request

CASES = Path('tests/fixtures/semantic_model_cases_20260914.json')
PACKETS = Path('tests/fixtures/semantic_model_responses_20260914.json')
URL = 'https://llmapi.paratera.com/v1'
FAKE_KEY = 'unit-test-placeholder-not-a-credential'


def model(handler, **kwargs):
    return OpenAICompatibleSemanticModel(base_url=URL, model='Kimi-K3', api_key=FAKE_KEY,
                                         transport=httpx.MockTransport(handler), **kwargs)


def request_and_packet():
    request = build_request(json.loads(CASES.read_text())['cases'][0]['messages'])
    packet = next(p for p in json.loads(PACKETS.read_text())['packets']
                  if p['prefix_sha256'] == request['prefix_sha256'])
    return request, packet


def envelope(packet, finish='stop'):
    return {'choices': [{'finish_reason': finish,
                         'message': {'content': json.dumps(packet)}}]}


def test_chat_protocol_and_payload_excludes_labels():
    request, packet = request_and_packet()

    def handler(req):
        assert str(req.url) == URL + '/chat/completions'
        assert req.headers['Authorization'] == 'Bearer ' + FAKE_KEY
        payload = json.loads(req.content)
        assert payload['model'] == 'Kimi-K3'
        assert payload['stream'] is False
        assert 'response_format' not in payload and 'thinking' not in payload
        user = json.loads(payload['messages'][1]['content'])
        assert set(user) == {'schema', 'prefix_sha256', 'visible_messages'}
        assert user['visible_messages'] == request['visible_messages']
        return httpx.Response(200, json=envelope(packet))

    client = model(handler)
    assert asyncio.run(client.extract(request)) == packet
    assert client.attempted_calls == 1
    assert FAKE_KEY not in json.dumps(client.provenance)


@pytest.mark.parametrize('status', [302, 401, 429, 500])
def test_http_failures_redacted_and_not_retried(status):
    client = model(lambda req: httpx.Response(status, text=FAKE_KEY,
                                             headers={'Location': 'https://example.com'}))
    with pytest.raises(SemanticAPIError, match=f'HTTP {status}') as exc:
        asyncio.run(client.extract(request_and_packet()[0]))
    assert FAKE_KEY not in str(exc.value)
    assert client.attempted_calls == 1


def test_timeout_is_redacted():
    def handler(req):
        raise httpx.ReadTimeout(FAKE_KEY, request=req)
    with pytest.raises(SemanticAPIError, match='timeout') as exc:
        asyncio.run(model(handler).extract(request_and_packet()[0]))
    assert FAKE_KEY not in str(exc.value)


def test_total_deadline_bounds_even_a_slow_transport():
    async def handler(req):
        await asyncio.sleep(1)
        pytest.fail('Deadline should cancel transport')
    client = model(handler, timeout=0.01)
    with pytest.raises(SemanticAPIError, match='total deadline'):
        asyncio.run(client.extract(request_and_packet()[0]))
    assert len(client.request_timings) == 1


def test_optional_vendor_thinking_mode_is_explicit_and_recorded():
    request, packet = request_and_packet()
    def handler(req):
        assert json.loads(req.content)['thinking'] == {'type': 'disabled'}
        return httpx.Response(200, json=envelope(packet))
    client = model(handler, thinking_mode='disabled')
    asyncio.run(client.extract(request))
    assert client.provenance['thinking_mode_requested'] == 'disabled'


def test_model_override_does_not_mutate_environment(monkeypatch, tmp_path):
    import os
    set_env(monkeypatch, tmp_path)
    client = OpenAICompatibleSemanticModel.from_env({'model': 'GLM-5.3-Flash', 'thinking_mode': 'disabled'})
    assert client.model == 'GLM-5.3-Flash' and client.thinking_mode == 'disabled'
    assert os.environ['TAU3_SEMANTIC_MODEL'] == 'Kimi-K3'


@pytest.mark.parametrize('body', [None, {'choices': []}, {'choices': [None]}, envelope({}, 'length'),
                                 {'choices': [{'finish_reason': 'stop', 'message': {'content': 'oops'}}]},
                                 envelope([])])
def test_invalid_or_truncated_envelope_abstains(body):
    with pytest.raises(SemanticAPIError):
        asyncio.run(model(lambda req: httpx.Response(200, json=body)).extract(request_and_packet()[0]))


def test_wrong_prefix_is_rejected_by_evidence_validator():
    request, packet = request_and_packet()
    packet['prefix_sha256'] = 'wrong'
    with pytest.raises(SemanticError):
        asyncio.run(model(lambda req: httpx.Response(200, json=envelope(packet))).extract(request))


@pytest.mark.parametrize('wrapper', ['```json\n{}\n```', '```\n{}\n```'])
def test_single_markdown_json_wrapper_is_accepted(wrapper):
    request, packet = request_and_packet()
    body = envelope(packet)
    body['choices'][0]['message']['content'] = wrapper.format(json.dumps(packet))
    assert asyncio.run(model(lambda req: httpx.Response(200, json=body)).extract(request)) == packet


def test_prose_around_markdown_packet_is_not_silently_discarded():
    request, packet = request_and_packet()
    body = envelope(packet)
    body['choices'][0]['message']['content'] = 'Explanation\n```json\n' + json.dumps(packet) + '\n```'
    with pytest.raises(SemanticAPIError):
        asyncio.run(model(lambda req: httpx.Response(200, json=body)).extract(request))


def test_malformed_event_type_is_rejected_safely():
    request, packet = request_and_packet()
    packet['events'][0]['kind'] = []
    with pytest.raises(SemanticAPIError, match='malformed field types'):
        asyncio.run(model(lambda req: httpx.Response(200, json=envelope(packet))).extract(request))


def set_env(monkeypatch, tmp_path, key=FAKE_KEY):
    monkeypatch.setenv('TAU3_ENV_FILE', str(tmp_path / 'missing.env'))
    monkeypatch.setenv('TAU3_SEMANTIC_BASE_URL', URL)
    monkeypatch.setenv('TAU3_SEMANTIC_MODEL', 'Kimi-K3')
    monkeypatch.setenv('TAU3_SEMANTIC_API_KEY', key)


def config():
    return yaml.safe_load(Path('configs/analysis/semantic_model_kimi_k3_20260914.yaml').read_text())


def test_missing_key_fails_before_network_or_output(monkeypatch, tmp_path):
    set_env(monkeypatch, tmp_path, key='')
    with pytest.raises(ValueError, match='TAU3_SEMANTIC_API_KEY'):
        asyncio.run(run(config(), tmp_path / 'out', transport=httpx.MockTransport(
            lambda req: pytest.fail('Unexpected network request'))))
    assert not (tmp_path / 'out').exists()


def test_two_case_audit_persists_packets_without_credentials(monkeypatch, tmp_path):
    set_env(monkeypatch, tmp_path)
    packets = {p['prefix_sha256']: p for p in json.loads(PACKETS.read_text())['packets']}

    def handler(req):
        prefix = json.loads(json.loads(req.content)['messages'][1]['content'])['prefix_sha256']
        return httpx.Response(200, json=envelope(packets[prefix]))

    output = tmp_path / 'out'
    summary = asyncio.run(run(config(), output, transport=httpx.MockTransport(handler)))
    assert summary['api_request_attempts'] == summary['valid_cases'] == 2
    assert summary['passed_pairs'] == summary['pairs'] == 1
    assert not summary['training_enabled']
    assert len((output / 'packets.jsonl').read_text().splitlines()) == 2
    assert all(FAKE_KEY not in p.read_text() for p in output.iterdir())


def test_audit_api_failure_is_abstention(monkeypatch, tmp_path):
    set_env(monkeypatch, tmp_path)
    summary = asyncio.run(run(config(), tmp_path / 'out', transport=httpx.MockTransport(
        lambda req: httpx.Response(401, text=FAKE_KEY))))
    assert summary['valid_cases'] == 0
    assert summary['checks'][0]['actual'] == 'abstain'
    assert summary['unavailable_pairs'] == 1
    assert all(FAKE_KEY not in p.read_text() for p in (tmp_path / 'out').iterdir())


def test_parallel_audit_shares_prefix_request_and_bounds_concurrency(monkeypatch, tmp_path):
    set_env(monkeypatch, tmp_path)
    dataset = json.loads(CASES.read_text())
    dataset['cases'] = dataset['cases'][:2] + [{**dataset['cases'][0], 'id': 'duplicate_prefix'}]
    case_path = tmp_path / 'cases.json'
    case_path.write_text(json.dumps(dataset))
    settings = {**config(), 'cases': str(case_path), 'limit': 3, 'concurrency': 2}
    packets = {p['prefix_sha256']: p for p in json.loads(PACKETS.read_text())['packets']}
    active = peak = 0

    async def handler(req):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        prefix = json.loads(json.loads(req.content)['messages'][1]['content'])['prefix_sha256']
        active -= 1
        return httpx.Response(200, json=envelope(packets[prefix]))

    summary = asyncio.run(run(settings, tmp_path / 'out', transport=httpx.MockTransport(handler)))
    assert summary['cases'] == summary['valid_cases'] == 3
    assert summary['api_request_attempts'] == peak == 2
    assert summary['provenance']['slot_schema'] == 'airline_slots_v1'
    assert summary['provenance']['slot_prompt_sha256']


def test_rejected_packet_is_saved_for_audit_without_becoming_valid(monkeypatch, tmp_path):
    set_env(monkeypatch, tmp_path)
    request, packet = request_and_packet()
    packet['prefix_sha256'] = 'wrong'
    summary = asyncio.run(run({**config(), 'limit': 1}, tmp_path / 'out',
                             transport=httpx.MockTransport(lambda req: httpx.Response(200, json=envelope(packet)))))
    assert summary['valid_cases'] == 0
    saved = json.loads((tmp_path / 'out' / 'packets.jsonl').read_text())
    assert saved['packet'] is None and saved['raw_packet'] == packet
    assert saved['error'] == 'prefix_mismatch'


def test_offline_replay_rejects_previously_unchecked_alias_amount(tmp_path):
    from tau3_grpo.analysis.replay_semantic_api import run as replay
    request, packet = request_and_packet()
    proposal = next(e for e in packet['events'] if e['kind'] == 'proposal')
    terms = proposal['data']['operations'][0]['terms']
    del terms['quoted_refund']
    terms['refund_amount'] = '999'
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'packets.jsonl').write_text(json.dumps({'prefix_sha256': request['prefix_sha256'],
                                                   'packet': packet, 'error': None}) + '\n')
    summary = replay({**config(), 'limit': 1}, source, tmp_path / 'out')
    assert summary['new_api_calls'] == summary['valid_cases'] == 0
    assert summary['errors']['approved'] == 'unsupported_money_role:quoted_refund'


def test_network_failure_is_not_a_correct_semantic_abstention(monkeypatch, tmp_path):
    set_env(monkeypatch, tmp_path)
    dataset = json.loads(CASES.read_text())
    dataset['cases'] = dataset['cases'][:2]
    dataset['pairs'] = [{'a': dataset['cases'][0]['id'], 'b': dataset['cases'][1]['id'],
                         'expected': 'abstain'}]
    path = tmp_path / 'cases.json'
    path.write_text(json.dumps(dataset))
    summary = asyncio.run(run({**config(), 'cases': str(path)}, tmp_path / 'out',
                             transport=httpx.MockTransport(lambda req: httpx.Response(500))))
    assert summary['passed_pairs'] == 0
    assert summary['unavailable_pairs'] == summary['abstained_pairs'] == 1


def test_supplement_only_replaces_missing_packets(tmp_path):
    from tau3_grpo.analysis.replay_semantic_api import run as replay
    request, packet = request_and_packet()
    source, extra = tmp_path / 'source', tmp_path / 'extra'
    source.mkdir()
    extra.mkdir()
    base = {'prefix_sha256': request['prefix_sha256'], 'packet': None, 'error': 'transport failed'}
    (source / 'packets.jsonl').write_text(json.dumps(base) + '\n')
    (extra / 'packets.jsonl').write_text(json.dumps({**base, 'packet': packet, 'error': None}) + '\n')
    result = replay({**config(), 'limit': 1}, source, tmp_path / 'out', supplements=[extra])
    assert result['valid_cases'] == 1 and result['new_api_calls'] == 0
    with pytest.raises(ValueError, match='only replace'):
        replay({**config(), 'limit': 1}, extra, tmp_path / 'not_created', supplements=[extra])
    assert not (tmp_path / 'not_created').exists()
