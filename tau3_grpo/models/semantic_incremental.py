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


def evidence_catalog(messages):
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
    return catalogue


def delta_request(messages, prior_packet, *, slot_schema):
    """Build a request from prior accepted events and the current visible prefix."""
    if not messages:
        raise SemanticError('incremental_empty_current_message')
    at = len(messages)-1
    if prior_packet['prefix_sha256'] != sha256_json(messages[:-1]):
        raise SemanticError('incremental_history_mismatch')
    request = build_request(messages, slot_schema=slot_schema)
    request['system'] += '\n' + PROMPT.read_text()
    catalogue = evidence_catalog(messages)
    payload = {'schema': 'semantic_delta_v1', 'prefix_sha256': request['prefix_sha256'],
               'current_at': at, 'current_message': messages[-1],
               'prior_events': [{k: deepcopy(v) for k, v in e.items() if k != 'evidence'}
                                for e in prior_packet['events']],
               'evidence_catalog': catalogue}
    if slot_schema == 'airline_slots_v4':
        reduced = compile_state(messages[:-1], prior_packet, task_id='incremental-validation',
                                db_hash='local', policy_hash='local', slot_schema=slot_schema,
                                include_references=True)
        payload['reference_state'] = reduced['references']
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
                 on_delta=None, saved_deltas=None, saved_provenance=None,
                 diagnostic_continue=False):
        if type(max_calls) is not int or max_calls <= 0:
            raise ValueError('max_incremental_calls must be a positive integer')
        if type(diagnostic_continue) is not bool:
            raise ValueError('diagnostic_continue must be boolean')
        self.client, self.slot_schema, self.max_calls = client, slot_schema, max_calls
        self.cache = {}
        self.response_packets = {}
        self.delta_records = []
        self.on_delta = on_delta
        self.saved_deltas = deepcopy(saved_deltas or {})
        self.saved_provenance = deepcopy(saved_provenance)
        self.reused_delta_count = 0
        self.diagnostic_continue = diagnostic_continue
        self.diagnostic_count = 0

    @property
    def provenance(self):
        return {**self.client.provenance, 'extraction_mode': 'incremental_v1',
                'max_incremental_calls': self.max_calls,
                'incremental_prompt_sha256': sha256_file(PROMPT),
                'history_rewrites': False,
                **({'diagnostic_continue': True, 'diagnostic_count': self.diagnostic_count}
                   if self.diagnostic_continue else {}),
                **({'saved_delta_source': self.saved_provenance,
                    'reused_delta_count': self.reused_delta_count} if self.saved_provenance else {})}

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
        try:
            prior = await self._prefix(messages[:-1])
        except (SemanticError, SemanticAPIError) as exc:
            if self.diagnostic_continue:
                await self._diagnose(messages, str(exc))
            raise
        current = messages[-1]
        if current.get('role') not in ('user', 'assistant') or not (current.get('content') or '').strip():
            # Tool knowledge is reduced locally, never interpreted as consent.
            packet = {**deepcopy(prior), 'prefix_sha256': sha256_json(messages)}
            compile_state(messages, packet, task_id='incremental-validation', db_hash='local',
                          policy_hash='local', slot_schema=self.slot_schema)
            return packet
        key = sha256_json(messages)
        reused = key in self.saved_deltas
        if not reused and self.attempted_calls >= self.max_calls:
            raise SemanticError('incremental_request_budget_exhausted')
        request, payload = delta_request(messages, prior, slot_schema=self.slot_schema)
        delta, error = None, None
        try:
            if reused:
                self.reused_delta_count += 1
                saved = self.saved_deltas[key]
                if saved.get('diagnostic_only'):
                    raise SemanticError('diagnostic_delta_cannot_resume_valid_state')
                if saved.get('at') != len(messages) - 1:
                    raise SemanticError('saved_delta_position_mismatch')
                if saved.get('delta') is None:
                    raise SemanticError('saved_request_failed_without_delta:no_retry')
                delta = deepcopy(saved['delta'])
            else:
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
            if self.saved_provenance:
                record['source'] = 'saved' if reused else 'new_request'
            self.delta_records.append(record)
            if self.on_delta is not None:
                self.on_delta(deepcopy(record))
        self.response_packets[request['prefix_sha256']] = deepcopy(packet)
        return packet

    async def _diagnose(self, messages, upstream_error):
        """Collect a suffix without fabricating a repaired, complete event ledger."""
        current = messages[-1]
        if current.get('role') not in ('user', 'assistant') or not (current.get('content') or '').strip():
            return
        if self.attempted_calls >= self.max_calls:
            return
        accepted_at, prior = 0, {'schema': SCHEMA, 'prefix_sha256': sha256_json([]), 'events': []}
        for length in range(len(messages) - 1, 0, -1):
            key = sha256_json(messages[:length])
            if key in self.response_packets:
                accepted_at, prior = length, deepcopy(self.response_packets[key])
                break
        request = build_request(messages, slot_schema=self.slot_schema)
        request['system'] += '\n' + PROMPT.read_text() + (
            '\nDIAGNOSTIC ONLY: an earlier event is unresolved. Interpret only the current message. '
            'Do not repair earlier events. reference_state is incomplete and valid only through '
            'reference_state_through_message. Your output cannot become a comparable state.')
        references = compile_state(messages[:accepted_at], prior, task_id='diagnostic', db_hash='local',
                                   policy_hash='local', slot_schema=self.slot_schema,
                                   include_references=True)['references']
        payload = {'schema': 'semantic_delta_v1', 'prefix_sha256': request['prefix_sha256'],
                   'current_at': len(messages) - 1, 'current_message': current,
                   'prior_events': [{k: v for k, v in e.items() if k != 'evidence'} for e in prior['events']],
                   'evidence_catalog': evidence_catalog(messages), 'reference_state': references,
                   'reference_state_through_message': accepted_at - 1,
                   'unresolved_message_indices': list(range(accepted_at, len(messages) - 1)),
                   'diagnostic_only': True}
        delta, api_error = None, None
        self.diagnostic_count += 1
        try:
            delta = await self.client.extract_json(request, user_payload=payload)
        except SemanticAPIError as exc:
            api_error = str(exc)
        finally:
            record = {'prefix_sha256': request['prefix_sha256'], 'at': len(messages) - 1,
                      'delta': delta, 'error': 'diagnostic_only:upstream_unresolved',
                      'upstream_error': upstream_error, 'api_error': api_error,
                      'diagnostic_only': True, 'eligible_for_state': False,
                      'reference_state_through_message': accepted_at - 1}
            self.delta_records.append(record)
            if self.on_delta is not None:
                self.on_delta(deepcopy(record))
