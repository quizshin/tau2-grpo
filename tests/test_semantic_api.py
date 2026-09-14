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
    assert all(FAKE_KEY not in p.read_text() for p in (tmp_path / 'out').iterdir())
