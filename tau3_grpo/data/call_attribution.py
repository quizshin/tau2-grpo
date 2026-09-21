"""Optional observations of call/token ownership; never modifies rewards or masks.

All offsets refer to actual emitted response IDs. Unsupported or ambiguous
formats remain observable but are never advertised as exact call credit.
"""

from __future__ import annotations

from copy import deepcopy

START, END = "<tool_call>", "</tool_call>"


def begin_call_attribution(token_ids, offset, tokenizer):
    ids = list(token_ids)
    result = {
        "schema": "tau3_call_attribution_v1",
        "coordinate_system": "response_token_offset_half_open",
        "emitted_span": [offset, offset + len(ids)],
        "emitted_token_ids": ids,
        "scan_status": "unavailable",
        "parse_status": "not_run",
        "blocks": [],
        "calls": [],
        "eligible_for_call_credit": False,
    }
    try:
        vocab = tokenizer.get_vocab()
        start_id, end_id = vocab[START], vocab[END]
        if (
            start_id == end_id
            or tokenizer.decode([start_id]) != START
            or tokenizer.decode([end_id]) != END
        ):
            raise ValueError("Tokenizer markers are not distinct atomic tokens")
    except (AttributeError, KeyError, TypeError, ValueError):
        result["unavailable_reason"] = "atomic_tool_markers_unavailable"
        return result
    depth, opening, ambiguous = 0, None, False
    for index, token in enumerate(ids):
        if token == start_id:
            if depth:
                ambiguous = True
            else:
                opening = index
            depth += 1
        elif token == end_id:
            if not depth:
                ambiguous = True
                continue
            depth -= 1
            if not depth:
                result["blocks"].append(
                    {
                        "block_index": len(result["blocks"]),
                        "emitted_span": [offset + opening, offset + index + 1],
                        "closure": "closed",
                    }
                )
    if depth:
        result["blocks"].append(
            {
                "block_index": len(result["blocks"]),
                "emitted_span": [offset + opening, offset + len(ids)],
                "closure": "incomplete",
            }
        )
    result["scan_status"] = "ambiguous" if ambiguous else "exact"
    return result


def attach_parser_sources(record, provenance, calls, tokenizer):
    """Bind sources returned by the SAME parser invocation, without re-parsing."""
    record["parse_status"] = provenance.get("status", "unsupported")
    sources = provenance.get("calls", [])
    source_blocks = provenance.get("blocks", [])
    blocks = record["blocks"]
    offset = record["emitted_span"][0]
    match = (
        record["scan_status"] == "exact"
        and len(blocks) == len(source_blocks)
        and len(sources) == len(calls)
        and provenance.get("status") == "parsed"
    )
    if match:
        for block, source in zip(blocks, source_blocks, strict=True):
            a, b = block["emitted_span"]
            # Decoding the original slice validates provenance; no re-encoding.
            if (
                tokenizer.decode(record["emitted_token_ids"][a - offset : b - offset])
                != source["text"]
            ):
                match = False
                break
    counts = {}
    for source in sources:
        index = source["block_index"]
        counts[index] = counts.get(index, 0) + 1
    for i, call in enumerate(calls):
        source = sources[i] if match else None
        block_index = source["block_index"] if source else None
        block = blocks[block_index] if block_index is not None else None
        alignment = "unavailable"
        if block is not None:
            alignment = (
                "ambiguous"
                if counts[block_index] != 1
                else "exact"
                if block["closure"] == "closed"
                else "incomplete"
            )
        record["calls"].append(
            {
                "call_index": i,
                "call_id": None,
                "name": call.name,
                "raw_arguments": call.arguments,
                "block_index": block_index,
                "alignment": alignment,
                "execution_status": "not_executed",
            }
        )


def bind_call_ids(record, recorded_calls):
    for item, call in zip(record["calls"], recorded_calls):
        item["call_id"] = getattr(call, "id", None)


def record_call_result(record, index, details):
    call = record["calls"][index]
    call.update(
        execution_status="executed",
        error=bool(details["error"]),
        dispatch_error=bool(details.get("dispatch_error", False)),
        db_hash_after=details.get("db_hash"),
    )


def record_call_start(record, index, interaction, request_id):
    call = record["calls"][index]
    call["execution_status"] = "started"
    snapshot = getattr(interaction, "tool_state_receipt", None)
    if not callable(snapshot):
        call["before_state_status"] = "unavailable"
        return
    try:
        call["db_hash_before"] = snapshot(request_id).get("db_hash")
        call["before_state_status"] = (
            "recorded" if call["db_hash_before"] is not None else "unavailable"
        )
    except Exception as error:
        # An optional observation must not change execution or invent a state.
        call["before_state_status"] = "unavailable"
        call["before_state_error_type"] = type(error).__name__


def retained_call_attribution(record, response_ids, response_mask):
    """Validate retained identity and publish a partition; clipping is explicit."""
    if len(response_ids) != len(response_mask) or any(
        value not in (0, 1) for value in response_mask
    ):
        raise ValueError("Actual retained token IDs and mask must align")
    result = deepcopy(record)
    start, end = result["emitted_span"]
    if end - start != len(result["emitted_token_ids"]) or start < 0 or end < start:
        raise ValueError("Invalid emitted call attribution span")
    limit = len(response_ids)
    a, b = min(start, limit), min(end, limit)
    if response_ids[a:b] != result["emitted_token_ids"][: b - a] or not all(response_mask[a:b]):
        raise ValueError("Call attribution differs from actual retained generated IDs/mask")
    result["retained_span"] = [a, b]
    result["tokens_discarded"] = end - start - (b - a)
    cursor, other = start, []
    for index, block in enumerate(result["blocks"]):
        x, y = block["emitted_span"]
        if block["block_index"] != index or not cursor <= x < y <= end:
            raise ValueError("Call blocks overlap or leave their assistant turn")
        if cursor < x:
            other.append([cursor, x])
        block["retained_span"] = [min(x, limit), min(y, limit)]
        block["tokens_discarded"] = y - x - (block["retained_span"][1] - block["retained_span"][0])
        cursor = y
    if cursor < end:
        other.append([cursor, end])
    result["other_generated_spans"] = other
    result["retained_other_generated_spans"] = [
        [min(x, limit), min(y, limit)] for x, y in other if min(x, limit) < min(y, limit)
    ]
    for call in result["calls"]:
        index = call["block_index"]
        block = result["blocks"][index] if index is not None else None
        call["eligible_for_call_credit"] = bool(
            call["alignment"] == "exact"
            and block is not None
            and not block["tokens_discarded"]
            and call["execution_status"] == "executed"
            and call["call_id"] is not None
        )
    result["eligible_for_call_credit"] = (
        bool(result["calls"])
        and len(result["blocks"]) == len(result["calls"])
        and all(call["eligible_for_call_credit"] for call in result["calls"])
    )
    return result
