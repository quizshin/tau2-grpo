"""Focused questions and explicit answers; v3 vocabulary stays immutable."""
from copy import deepcopy

from tau3_grpo.algorithms.anchors import semantic_slots_v3 as v3
from tau3_grpo.algorithms.anchors.semantic_slots import fail
from tau3_grpo.algorithms.anchors.semantic_questions import normalize_question

VERSION = 'airline_slots_v4'


def normalize_packet(packet, messages):
    packet = deepcopy(packet)
    if not isinstance(packet, dict) or not isinstance(packet.get('events'), list):
        fail('invalid_slot_packet')
    normalized = []
    for event in packet['events']:
        if not isinstance(event, dict) or not isinstance(event.get('data'), dict):
            fail('invalid_slot_event')
        if event.get('kind') == 'question':
            event['data'] = normalize_question(event['data'], event, messages)
        elif event.get('kind') == 'question_answer':
            data = event['data']
            if (set(data) != {'question_id', 'value'} or not isinstance(data['question_id'], str)
                    or not isinstance(data['value'], str)):
                fail('invalid_question_answer')
        else:
            event = v3.normalize_packet({**packet, 'events': [event]}, messages)['events'][0]
        normalized.append(event)
    return {**packet, 'events': normalized}
