import copy

import pytest
from test_qwen35 import MESSAGES, TOOLS, tokenizer

from tau3_grpo.models.qwen35_template import thinking_options
from tau3_grpo.training.sft.dataset import build_supervised_example


def test_thinking_masks_and_history():
    tok = tokenizer('Qwen3.5-4B')
    messages = copy.deepcopy(MESSAGES)
    for i, m in enumerate(messages):
        if m['role'] == 'assistant':
            m['reasoning'] = f'REASONING_SENTINEL_{i}'
    original = copy.deepcopy(messages)
    options = dict(enable_thinking=True, supervise_reasoning=True,
                   preserve_historical_reasoning=True)
    result = build_supervised_example(messages, tok, tools=TOOLS, **options)
    labeled = tok.decode([v for v in result['labels'] if v != -100])
    for m in messages:
        if m['role'] == 'assistant':
            assert m['reasoning'] in labeled
    for text in ['SYSTEM_ONLY', 'CUSTOMER_ONLY', 'OBSERVATION_ONLY', 'CUSTOMER_CONFIRMATION_ONLY']:
        assert text not in labeled
    assert messages == original
    no_history = build_supervised_example(messages, tok, tools=TOOLS,
        enable_thinking=True, supervise_reasoning=True)
    assert no_history['n_total_tokens'] < result['n_total_tokens']
    no_supervision = build_supervised_example(messages, tok, tools=TOOLS,
        enable_thinking=True, preserve_historical_reasoning=True)
    assert no_supervision['input_ids'] == result['input_ids']
    assert 'REASONING_SENTINEL' not in tok.decode([v for v in no_supervision['labels'] if v != -100])
    with pytest.raises(ValueError, match='exceeding'):
        build_supervised_example(messages, tok, tools=TOOLS, max_length=10, **options)


def test_off_defaults_and_missing_reasoning():
    tok = tokenizer('Qwen3.5-4B')
    assert build_supervised_example(MESSAGES, tok, tools=TOOLS) == build_supervised_example(
        MESSAGES, tok, tools=TOOLS, **thinking_options({}))
    result = build_supervised_example(MESSAGES, tok, tools=TOOLS,
        enable_thinking=True, supervise_reasoning=True, preserve_historical_reasoning=True)
    assert result['n_label_tokens'] > 0


@pytest.mark.parametrize('config', [dict(enable_thinking='false'), dict(supervise_reasoning=True),
    dict(preserve_historical_reasoning=True)])
def test_invalid_thinking_configuration(config):
    with pytest.raises(ValueError):
        thinking_options(config)


@pytest.mark.parametrize('indices', [[3], [5, 7], [3, 7]])
def test_approved_targets_keep_native_history(indices):
    tok = tokenizer('Qwen3.5-4B')
    messages = copy.deepcopy(MESSAGES)
    messages[3]['tool_calls'] *= 2
    full = build_supervised_example(messages, tok, tools=TOOLS)
    masked = build_supervised_example(messages, tok, tools=TOOLS, approved_indices=indices)
    assert full['input_ids'] == masked['input_ids']
    labeled = tok.decode([v for v in masked['labels'] if v != -100])
    assert 'Welcome.' not in labeled
    assert ('get_user_details' in labeled) == (3 in indices)
    assert ('Please confirm.' in labeled) == (5 in indices)
    assert ('Done.' in labeled) == (7 in indices)
    assert labeled.count('<|im_end|>') == len(indices)
    for text in ['SYSTEM_ONLY', 'CUSTOMER_ONLY', 'OBSERVATION_ONLY', 'CUSTOMER_CONFIRMATION_ONLY']:
        assert text not in labeled
    assert 0 < masked['n_label_tokens'] < full['n_label_tokens']


@pytest.mark.parametrize('indices', [[], [0], [2], [3, 3], [True], [-1], [99], '3'])
def test_invalid_approved_targets_fail_closed(indices):
    from tau3_grpo.models.qwen35_template import approved_assistant_indices
    with pytest.raises(ValueError, match='approved assistant'):
        approved_assistant_indices(MESSAGES, indices)


def test_dataset_requires_and_preserves_target_contract(tmp_path):
    import json

    from tau3_grpo.training.sft.dataset import TrajectorySFTDataset
    tok = tokenizer('Qwen3.5-4B')
    path = tmp_path / 'data.jsonl'
    record = {'messages': MESSAGES}
    path.write_text(json.dumps(record) + '\n')
    with pytest.raises(ValueError, match='supervision contract'):
        TrajectorySFTDataset(path, tok, tools=TOOLS, require_approved_targets=True)
    record['supervision'] = {'version': 'approved_assistant_v1', 'message_indices': [3, 7]}
    path.write_text(json.dumps(record) + '\n')
    data = TrajectorySFTDataset(path, tok, tools=TOOLS, require_approved_targets=True)
    labeled = tok.decode([v for v in data.examples[0]['labels'] if v != -100])
    assert 'Welcome.' not in labeled and 'Please confirm.' not in labeled
    assert 'get_user_details' in labeled and 'Done.' in labeled
    out = tmp_path / 'effective.jsonl'
    data.write_effective_jsonl(out)
    assert json.loads(out.read_text())['supervision'] == record['supervision']
