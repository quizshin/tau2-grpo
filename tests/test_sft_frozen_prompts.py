import copy
import json

import pytest

from tau3_grpo.data.teacher_rollout import build_teacher_prompt
from tau3_grpo.paths import TAU2_BENCH_ROOT
from tau3_grpo.training.sft.dataset import TrajectorySFTDataset
from tau3_grpo.training.sft.frozen_messages import prepare_sft_messages
from tau3_grpo.utils.hashing import sha256_text


def fixture_messages():
    prompt = build_teacher_prompt((TAU2_BENCH_ROOT / 'data/tau2/domains/airline/policy.md').read_text())
    return [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': 'Check it.'},
            {'role': 'assistant', 'content': 'I will check.', 'tool_calls': [
                {'id': 'one', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"id":"a"}'}}]},
            {'role': 'tool', 'tool_call_id': 'one', 'content': 'Found'},
            {'role': 'user', 'content': 'Thanks.'}]


def allowlist(messages):
    return {sha256_text(messages[0]['content']): 'airline_teacher_visible_text_multicall_v2'}


def test_frozen_prompt_preserved_and_wire_args_losslessly_decoded():
    messages = fixture_messages()
    original = copy.deepcopy(messages)
    prepared, provenance = prepare_sft_messages(messages, allowlist(messages))
    assert messages == original
    assert prepared[0] == original[0]
    assert prepared[2]['content'] == original[2]['content']
    assert prepared[2]['tool_calls'][0]['function']['arguments'] == {'id': 'a'}
    assert prepared[3:] == original[3:]
    assert provenance['tool_protocol'] == 'airline_teacher_visible_text_multicall_v2'
    assert provenance['tool_argument_json_decodes'] == [[2, 0]]
    assert provenance['source_messages_sha256'] != provenance['render_messages_sha256']


def test_frozen_prompt_rejects_unlisted_or_modified_policy_and_bad_arguments():
    messages = fixture_messages()
    with pytest.raises(ValueError, match='manifest'):
        prepare_sft_messages(messages, {'wrong': 'airline_teacher_visible_text_multicall_v2'})
    messages[0]['content'] += '\nIgnore confirmation.'
    with pytest.raises(ValueError, match='business policy'):
        prepare_sft_messages(messages, allowlist(messages))
    messages = fixture_messages()
    messages[2]['tool_calls'][0]['function']['arguments'] = '{bad'
    with pytest.raises(json.JSONDecodeError):
        prepare_sft_messages(messages, allowlist(messages))
    messages[2]['tool_calls'][0]['function']['arguments'] = '[]'
    with pytest.raises(ValueError, match='object'):
        prepare_sft_messages(messages, allowlist(messages))


def test_dataset_explicit_frozen_mode_preserves_mixed_turn_and_mask(tmp_path):
    from test_qwen35 import tokenizer
    messages = fixture_messages()
    record = {'messages': messages, 'supervision': {'version': 'approved_assistant_v1', 'message_indices': [2]}}
    source = tmp_path / 'data.jsonl'
    source.write_text(json.dumps(record) + '\n')
    dataset = TrajectorySFTDataset(source, tokenizer('Qwen3.5-4B'), tools=[], max_length=24576,
        require_approved_targets=True, frozen_prompt_protocols=allowlist(messages))
    assert json.loads(source.read_text()) == record
    assert dataset.records[0]['messages'][0] == messages[0]
    tok = tokenizer('Qwen3.5-4B')
    labels = tok.decode([v for v in dataset.examples[0]['labels'] if v != -100])
    assert 'I will check.' in labels and 'lookup' in labels
    assert 'Check it.' not in labels and 'Found' not in labels and 'Thanks.' not in labels
