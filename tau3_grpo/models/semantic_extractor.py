"""Semantic request construction and an explicit, simulated fixture provider."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Protocol

from tau3_grpo.algorithms.anchors.semantic_state import SCHEMA, validate_packet
from tau3_grpo.analysis.replay_decisions import visible_message
from tau3_grpo.utils.hashing import sha256_json, sha256_file

PROMPT_PATH = Path(__file__).resolve().parents[2] / 'configs/prompts/semantic_events_v1.txt'


def slot_prompt_path(version):
    if version not in ('airline_slots_v1', 'airline_slots_v2'):
        raise ValueError('Unsupported slot schema')
    return PROMPT_PATH.with_name('semantic_' + version + '.txt')


class SemanticModel(Protocol):
    async def extract(self, request: dict) -> dict: ...


def build_request(messages, *, slot_schema=None):
    prefix = [visible_message(m) for m in messages]
    request = {'system': PROMPT_PATH.read_text(), 'schema': SCHEMA,
            'prefix_sha256': sha256_json(prefix), 'visible_messages': prefix}
    if slot_schema is not None:
        request['slot_schema'] = slot_schema
        request['system'] += '\n' + slot_prompt_path(slot_schema).read_text()
    return request


class FixtureSemanticModel:
    """Pre-authored simulated outputs, selected ONLY by input prefix hash.

    No pair labels, rewards or expected decisions are passed to this provider.
    Missing fixtures fail, rather than silently falling back to an empty state.
    """
    simulated = True

    def __init__(self, path):
        self.path = Path(path)
        data = json.loads(self.path.read_text())
        if data.get('simulated') is not True:
            raise ValueError('Fixture file must declare simulated=true')
        self.packets = {}
        for packet in data['packets']:
            key = packet['prefix_sha256']
            if key in self.packets:
                raise ValueError('Duplicate fixture prefix')
            self.packets[key] = packet
        self.fingerprint = sha256_file(self.path)

    async def extract(self, request):
        prefix = request['visible_messages']
        if request['prefix_sha256'] != sha256_json(prefix):
            raise ValueError('Model request prefix changed')
        key = request['prefix_sha256']
        if key not in self.packets:
            raise LookupError('No simulated model output for this exact prefix')
        packet = deepcopy(self.packets[key])
        validate_packet(prefix, packet)
        return packet
