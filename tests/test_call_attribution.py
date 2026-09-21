"""Real tokenizer fixtures and isolated native parser; no model/GPU required."""

import ast
import asyncio
import json
import logging
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import regex
from tokenizers import Tokenizer

from tau3_grpo.data.call_attribution import (
    attach_parser_sources,
    begin_call_attribution,
    bind_call_ids,
    record_call_result,
    retained_call_attribution,
)
from tau3_grpo.data.trajectory import trajectory_facts

ROOT = Path(__file__).resolve().parents[1]


class RealTokenizer:
    def __init__(self):
        self.backend = Tokenizer.from_file(str(ROOT / "models/Qwen3.5-4B/tokenizer.json"))

    def encode(self, text):
        return self.backend.encode(text, add_special_tokens=False).ids

    def decode(self, ids, **kwargs):
        return self.backend.decode(ids, skip_special_tokens=False)

    def get_vocab(self):
        return self.backend.get_vocab()


def isolated_parser(tokenizer, path=None):
    path = path or ROOT / "verl/verl/experimental/agent_loop/tool_parser.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen3XMLToolParser"
    )
    cls.decorator_list = []

    class ParserBase:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    env = dict(
        ast=ast,
        json=json,
        regex=regex,
        logger=logging.getLogger(__name__),
        Optional=Optional,
        Any=Any,
        OpenAIFunctionToolSchema=object,
        FunctionCall=SimpleNamespace,
        ToolParser=ParserBase,
        rollout_trace_op=lambda f: f,
        get_event_loop=asyncio.get_running_loop,
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"
        ),
        env,
    )
    return env["Qwen3XMLToolParser"](tokenizer)


@pytest.fixture(scope="module")
def tokenizer():
    return RealTokenizer()


def xml(value="1+1", name="calculate"):
    return f"<tool_call><function={name}><parameter=expression>{value}</parameter></function></tool_call>"


async def observe(text, tokenizer, offset=0):
    ids = tokenizer.encode(text)  # Synthetic model-output fixture, never historic token recovery.
    record = begin_call_attribution(ids, offset, tokenizer)
    content, calls, source = await isolated_parser(tokenizer).extract_tool_calls_with_provenance(
        ids
    )
    attach_parser_sources(record, source, calls, tokenizer)
    return ids, record, content, calls


def execute(record):
    bind_call_ids(record, [SimpleNamespace(id=f"call-{i}") for i in range(len(record["calls"]))])
    for i in range(len(record["calls"])):
        record_call_result(record, i, {"error": i == 1, "db_hash": f"state-{i}"})


def test_duplicate_calls_chinese_prose_and_eos_partition(tokenizer):
    text = "先查询。" + xml("中文参数") + "中间说明" + xml("中文参数") + "完成。<|im_end|>"
    ids, record, content, calls = asyncio.run(observe(text, tokenizer, offset=3))
    assert len(calls) == 2 and calls[0].arguments == calls[1].arguments
    execute(record)
    result = retained_call_attribution(record, [8, 9, 10] + ids, [0, 0, 0] + [1] * len(ids))
    assert result["eligible_for_call_credit"]
    assert [x["call_id"] for x in result["calls"]] == ["call-0", "call-1"]
    assert [x["error"] for x in result["calls"]] == [False, True]
    cover = [0] * len(ids)
    for a, b in result["other_generated_spans"] + [x["emitted_span"] for x in result["blocks"]]:
        for i in range(a - 3, b - 3):
            cover[i] += 1
    assert cover == [1] * len(ids)
    for block in result["blocks"]:
        a, b = block["emitted_span"]
        assert tokenizer.decode(ids[a - 3 : b - 3]) == xml("中文参数")
    assert content == "先查询。中间说明完成。<|im_end|>"


@pytest.mark.parametrize(
    "text,alignment,execution_count",
    [
        (xml().removesuffix("</tool_call>"), "incomplete", 1),
        ("<tool_call><function=a></function><function=b></function></tool_call>", "ambiguous", 2),
        ("<tool_call>" + xml() + "</tool_call>", "unavailable", 1),
        ("</tool_call>" + xml(), "unavailable", 1),
    ],
)
def test_uncertain_boundaries_do_not_change_parser_or_certify_credit(
    tokenizer, text, alignment, execution_count
):
    ids, record, content, calls = asyncio.run(observe(text, tokenizer))
    baseline_content, baseline_calls = asyncio.run(
        isolated_parser(tokenizer).extract_tool_calls(ids)
    )
    assert content == baseline_content and calls == baseline_calls
    assert len(calls) == execution_count
    assert all(c["alignment"] == alignment for c in record["calls"])
    execute(record)
    assert not retained_call_attribution(record, ids, [1] * len(ids))["eligible_for_call_credit"]


@pytest.mark.parametrize(
    "text", ["普通文字", "<tool_call>not a function</tool_call>", "<tool_call><function=broken"]
)
def test_no_call_or_parse_failure_has_no_fabricated_id(tokenizer, text):
    ids, record, _, calls = asyncio.run(observe(text, tokenizer))
    assert calls == [] and record["calls"] == []
    result = retained_call_attribution(record, ids, [1] * len(ids))
    assert not result["eligible_for_call_credit"]
    if "broken" in text:
        assert result["parse_status"] == "parse_failed"


def test_budget_stop_before_parser_preserves_unexecuted_raw_block(tokenizer):
    ids = tokenizer.encode(xml())
    record = begin_call_attribution(ids, 0, tokenizer)
    result = retained_call_attribution(record, ids, [1] * len(ids))
    assert len(result["blocks"]) == 1 and result["parse_status"] == "not_run"
    assert not result["calls"] and not result["eligible_for_call_credit"]


@pytest.mark.parametrize("keep", [0, 1, -1])
def test_retention_does_not_promote_partial_or_discarded_calls(tokenizer, keep):
    ids, record, _, _ = asyncio.run(observe(xml(), tokenizer))
    execute(record)
    count = len(ids) - 1 if keep == -1 else keep
    result = retained_call_attribution(record, ids[:count], [1] * count)
    assert result["emitted_token_ids"] == ids
    assert result["tokens_discarded"] == len(ids) - count
    assert not result["eligible_for_call_credit"]


def test_mask_token_and_turn_tampering_rejected(tokenizer):
    ids, record, _, _ = asyncio.run(observe(xml(), tokenizer))
    with pytest.raises(ValueError):
        retained_call_attribution(record, [0] + ids[1:], [1] * len(ids))
    with pytest.raises(ValueError):
        retained_call_attribution(record, ids, [0] + [1] * (len(ids) - 1))
    turn = {"token_span": [0, len(ids)], "generated_logprobs_available": True}
    a = trajectory_facts(
        request_id="x",
        task_id="t",
        turns=[turn],
        response_ids=ids,
        response_mask=[1] * len(ids),
        call_attributions=[record],
    )
    assert a["turns"][0]["call_attribution"]["emitted_token_ids"] == ids
    bad = deepcopy(record)
    bad["emitted_span"][0] += 1
    with pytest.raises(ValueError):
        trajectory_facts(
            request_id="x",
            task_id="t",
            turns=[turn],
            response_ids=ids,
            response_mask=[1] * len(ids),
            call_attributions=[bad],
        )


def test_parser_sources_do_not_leak_across_concurrent_requests(tokenizer):
    async def run():
        parser = isolated_parser(tokenizer)
        results = await asyncio.gather(
            *(
                parser.extract_tool_calls_with_provenance(tokenizer.encode(xml(str(i))))
                for i in range(12)
            )
        )
        for i, (_, calls, sources) in enumerate(results):
            assert json.loads(calls[0].arguments)["expression"] == str(i)
            assert sources["blocks"][0]["text"] == xml(str(i))

    asyncio.run(run())


def test_unavailable_markers_are_observations_not_rollout_errors(tokenizer):
    ids = tokenizer.encode(xml())
    record = begin_call_attribution(ids, 0, object())
    assert record["scan_status"] == "unavailable"
    assert retained_call_attribution(record, ids, [1] * len(ids))["other_generated_spans"] == [
        [0, len(ids)]
    ]


def test_empty_block_does_not_hide_later_valid_call(tokenizer):
    ids, record, content, calls = asyncio.run(observe("<tool_call></tool_call>" + xml(), tokenizer))
    assert len(calls) == 1 and content == ""
    assert record["calls"][0]["block_index"] == 1 and record["calls"][0]["alignment"] == "exact"
    execute(record)
    result = retained_call_attribution(record, ids, [1] * len(ids))
    assert result["calls"][0]["eligible_for_call_credit"]
    assert not result["eligible_for_call_credit"]  # Earlier unparsed block remains visible.


def test_unexecuted_tail_does_not_gain_credit_from_executed_prefix(tokenizer):
    ids, record, _, _ = asyncio.run(observe(xml() + xml("2+2"), tokenizer))
    bind_call_ids(record, [SimpleNamespace(id="first")])
    record_call_result(record, 0, {"error": False, "db_hash": "state"})
    result = retained_call_attribution(record, ids, [1] * len(ids))
    assert result["calls"][0]["eligible_for_call_credit"]
    assert not result["calls"][1]["eligible_for_call_credit"]
    assert result["calls"][1]["execution_status"] == "not_executed"


def test_optional_state_snapshot_failure_does_not_modify_execution_record(tokenizer):
    from tau3_grpo.data.call_attribution import record_call_start

    ids, record, _, _ = asyncio.run(observe(xml(), tokenizer))

    def fail(request_id):
        raise RuntimeError("unavailable")

    record_call_start(record, 0, SimpleNamespace(tool_state_receipt=fail), "session")
    assert record["calls"][0]["before_state_status"] == "unavailable"
    assert record["calls"][0]["execution_status"] == "started"
    assert not retained_call_attribution(record, ids, [1] * len(ids))["eligible_for_call_credit"]
