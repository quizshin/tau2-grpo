import copy
import pytest
from test_qwen35 import tokenizer, MESSAGES, TOOLS
from tau3_grpo.training.sft.dataset import build_supervised_example
from tau3_grpo.models.qwen35_template import thinking_options


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
