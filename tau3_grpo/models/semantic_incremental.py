"""Append-only, evidence-ID semantic extraction; never invoked by the RL hook."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path

from tau3_grpo.algorithms.anchors.semantic_state import SCHEMA, SemanticError, compile_state
from tau3_grpo.models.semantic_api import SemanticAPIError
from tau3_grpo.models.semantic_extractor import build_request
from tau3_grpo.utils.hashing import sha256_file, sha256_json

PROMPT = Path(__file__).resolve().parents[2] / 'configs/prompts/semantic_incremental_v1.txt'


def delta_request(messages, prior_packet, *, slot_schema):
    """Build a request from prior accepted events and the current visible prefix."""
    if not messages:
        raise SemanticError('incremental_empty_current_message')
    at = len(messages)-1
    if prior_packet['prefix_sha256'] != sha256_json(messages[:-1]):
        raise SemanticError('incremental_history_mismatch')
    request = build_request(messages, slot_schema=slot_schema)
    request['system'] += '\n' + PROMPT.read_text()
    catalogue = {}
    for index, message in enumerate(messages):
        content = message.get('content')
        if message.get('role') in ('user', 'assistant') and isinstance(content, str) and content.strip():
            catalogue[f'm{index}'] = {'message_index': index, 'start': 0,
                                     'end': len(content), 'quote': content}
    # Only successful, ordered, matched observed facts may support a claim.
    from tau3_grpo.algorithms.anchors.grounded import extract
    for index, fact in enumerate(extract(messages)['facts']):
        catalogue[f'f{index}'] = {'message_index': fact['evidence']['message_index'],
                                  'pointer': fact['evidence']['pointer'], 'value': fact['value']}
    payload = {'schema': 'semantic_delta_v1', 'prefix_sha256': request['prefix_sha256'],
               'current_at': at, 'current_message': messages[-1],
               'prior_events': [{k: deepcopy(v) for k, v in e.items() if k != 'evidence'}
                                for e in prior_packet['events']],
               'evidence_catalog': catalogue}
    return request, payload


def append_delta(messages, prior_packet, delta, catalogue, *, slot_schema):
    """Resolve local evidence IDs and atomically validate the entire new state."""
    at = len(messages)-1
    if prior_packet['prefix_sha256'] != sha256_json(messages[:-1]):
        raise SemanticError('incremental_history_mismatch')
    if (not isinstance(delta, dict) or set(delta) != {'schema', 'prefix_sha256', 'events'}
            or delta['schema'] != 'semantic_delta_v1' or delta['prefix_sha256'] != sha256_json(messages)
            or not isinstance(delta['events'], list)):
        raise SemanticError('incremental_packet_schema')
    events = deepcopy(prior_packet['events'])
    for row in delta['events']:
        if (not isinstance(row, dict) or set(row) != {'id', 'kind', 'data', 'evidence'}
                or not isinstance(row['id'], str) or not row['id'].startswith(f'm{at}:')
                or not isinstance(row['evidence'], list) or not row['evidence']
                or any(not isinstance(ref, str) or ref not in catalogue for ref in row['evidence'])):
            raise SemanticError('incremental_event_schema_or_reference')
        event = deepcopy(row)
        event['at'] = at
        event['evidence'] = [deepcopy(catalogue[ref]) for ref in row['evidence']]
        if event['kind'] == 'context':
            if event['data'] != {}:
                raise SemanticError('incremental_context_must_be_local')
            event['data'] = {'text': messages[at]['content']}
        events.append(event)
    packet = {'schema': SCHEMA, 'prefix_sha256': sha256_json(messages), 'events': events}
    # Reducer checks references/active proposals after every append, so later
    # messages cannot silently repair an earlier invalid consent or unknown.
    compile_state(messages, packet, task_id='incremental-validation', db_hash='local',
                  policy_hash='local', slot_schema=slot_schema)
    return packet


class IncrementalSemanticModel:
    """Per-audit prefix cache; concurrent branches cannot mutate each other's log."""
    def __init__(self, client, *, slot_schema='airline_slots_v2', max_calls=32,
                 on_delta=None):
        if type(max_calls) is not int or max_calls <= 0:
            raise ValueError('max_incremental_calls must be a positive integer')
        self.client, self.slot_schema, self.max_calls = client, slot_schema, max_calls
        self.cache = {}
        self.response_packets = {}
        self.delta_records = []
        self.on_delta = on_delta

    @property
    def provenance(self):
        return {**self.client.provenance, 'extraction_mode': 'incremental_v1',
                'max_incremental_calls': self.max_calls,
                'incremental_prompt_sha256': sha256_file(PROMPT),
                'history_rewrites': False}

    @property
    def attempted_calls(self):
        return self.client.attempted_calls

    @property
    def metadata(self):
        return self.client.metadata

    @property
    def request_timings(self):
        return self.client.request_timings

    async def extract(self, request):
        messages = request['visible_messages']
        if request['prefix_sha256'] != sha256_json(messages) or request.get('slot_schema') != self.slot_schema:
            raise SemanticError('incremental_request_mismatch')
        packet = deepcopy(await self._prefix(messages))
        self.response_packets[request['prefix_sha256']] = deepcopy(packet)
        return packet

    async def _prefix(self, messages):
        key = sha256_json(messages)
        if key not in self.cache:
            self.cache[key] = asyncio.create_task(self._advance(deepcopy(messages)))
        return await self.cache[key]

    async def _advance(self, messages):
        if not messages:
            return {'schema': SCHEMA, 'prefix_sha256': sha256_json([]), 'events': []}
        prior = await self._prefix(messages[:-1])
        current = messages[-1]
        if current.get('role') not in ('user', 'assistant') or not (current.get('content') or '').strip():
            # Tool knowledge is reduced locally, never interpreted as consent.
            packet = {**deepcopy(prior), 'prefix_sha256': sha256_json(messages)}
            compile_state(messages, packet, task_id='incremental-validation', db_hash='local',
                          policy_hash='local', slot_schema=self.slot_schema)
            return packet
        if self.attempted_calls >= self.max_calls:
            raise SemanticError('incremental_request_budget_exhausted')
        request, payload = delta_request(messages, prior, slot_schema=self.slot_schema)
        delta, error = None, None
        try:
            delta = await self.client.extract_json(request, user_payload=payload)
            packet = append_delta(messages, prior, delta, payload['evidence_catalog'], slot_schema=self.slot_schema)
        except (TypeError, KeyError, IndexError, AttributeError):
            error = 'incremental_malformed_field_types'
            raise SemanticError(error) from None
        except (SemanticError, SemanticAPIError) as exc:
            error = str(exc)
            raise
        finally:
            record = {'prefix_sha256': request['prefix_sha256'],
                      'at': len(messages)-1, 'delta': delta, 'error': error}
            self.delta_records.append(record)
            if self.on_delta is not None:
                self.on_delta(deepcopy(record))
        self.response_packets[request['prefix_sha256']] = deepcopy(packet)
        return packet
