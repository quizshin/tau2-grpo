"""Agent prompt compatibility across source data, cached data and evaluation."""

from __future__ import annotations

import copy
import hashlib
import json

from tau3_grpo.paths import TAU2_BENCH_ROOT
from tau3_grpo.prompts import (
    SINGLE_CALL_RULE,
    TOOL_PROTOCOL_VERSION,
    build_system_prompt,
    prepare_agent_messages,
    prompt_provenance,
)


def test_multicall_changes_only_the_official_call_restriction():
    path = TAU2_BENCH_ROOT / "data/tau2/domains/airline/policy.md"
    before = path.read_bytes()
    policy = before.decode()
    prompt = build_system_prompt()
    embedded = prompt.split("<policy>\n", 1)[1].removesuffix("\n</policy>")
    assert embedded == policy.replace(SINGLE_CALL_RULE, "").strip()
    assert "obtain explicit user confirmation (yes)" in embedded
    assert "one or more tool calls" in prompt
    assert "sequentially in the order listed" in prompt
    assert "dependent call in a later turn" in prompt
    assert SINGLE_CALL_RULE not in prompt
    assert path.read_bytes() == before


def test_cached_sft_and_rl_prompts_are_replaced_without_rewriting_trajectory():
    messages = [
        {"role": "system", "content": "Make exactly ONE tool call. "
         "You CANNOT make multiple tool calls. LLM JUDGE SHOULD NOT CARE OF THIS."},
        {"role": "user", "content": "Please check my flights."},
        {"role": "assistant", "content": "", "reasoning": "Inspect both flights.",
         "tool_calls": [{"name": "lookup", "arguments": {"id": "a"}},
                        {"name": "lookup", "arguments": {"id": "b"}}]},
        {"role": "tool", "content": "first result"},
        {"role": "tool", "content": "second result"},
    ]
    original = copy.deepcopy(messages)
    prepared = prepare_agent_messages(messages)
    assert prepared[0]["content"] == build_system_prompt()
    assert prepared[1:] == original[1:]
    assert messages == original
    assert prepare_agent_messages(prepared) == prepared
    assert "LLM JUDGE" not in prepared[0]["content"]
    assert "exactly ONE" not in prepared[0]["content"]


def test_eval_agent_and_training_share_the_same_system_prompt(requires_tau2):
    from tau3_grpo.envs.agent import MultiCallAirlineAgent

    policy = (TAU2_BENCH_ROOT / "data/tau2/domains/airline/policy.md").read_text()
    agent = MultiCallAirlineAgent(tools=[], domain_policy=policy, llm="openai/not-called")
    assert agent.get_init_state().system_messages[0].content == build_system_prompt()
    assert prompt_provenance()["tool_protocol"] == TOOL_PROTOCOL_VERSION
    assert prompt_provenance()["benchmark_policy_modified"] is True
    assert prompt_provenance()["agent_system_prompt_sha256"] == hashlib.sha256(
        agent.system_prompt.encode()
    ).hexdigest()


def test_sft_loads_legacy_file_with_effective_prompt_and_exports_it(tmp_path):
    from tau3_grpo.training.sft.dataset import TrajectorySFTDataset

    class Tokenizer:
        name_or_path = "test"

        def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt):
            text = "".join(f"<{m['role']}>{m['content']}<end>" for m in messages)
            if add_generation_prompt:
                text += "<assistant>"
            return list(text.encode())

    raw = {"messages": [{"role": "system", "content": "only ONE tool call"},
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "How can I help?"}],
           "metadata": {"dialogue_hash": "original-hash"}}
    source = tmp_path / "legacy.jsonl"
    source.write_text(json.dumps(raw) + "\n")
    before = source.read_bytes()
    dataset = TrajectorySFTDataset(source, Tokenizer(), max_length=100000)
    output = tmp_path / "effective.jsonl"
    dataset.write_effective_jsonl(output)
    effective = json.loads(output.read_text())
    assert source.read_bytes() == before
    assert effective["messages"][0]["content"] == build_system_prompt()
    assert effective["messages"][1:] == raw["messages"][1:]
    assert effective["metadata"]["source_dialogue_hash"] == "original-hash"
    assert effective["metadata"]["tool_protocol"] == TOOL_PROTOCOL_VERSION
    labels = dataset.examples[0]["labels"]
    supervised = bytes(token for token in labels if token != -100).decode()
    assert supervised == "How can I help?<end>"
