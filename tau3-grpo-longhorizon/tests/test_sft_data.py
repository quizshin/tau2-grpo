"""AReaL complete-dialogue selection and assistant-only SFT masking."""

from __future__ import annotations

import json

from tau3_grpo.data.sft import (
    load_complete_airline_dialogues,
    reason_similarity,
    select_dialogues,
    write_dialogue_split,
)
from tau3_grpo.sft.dataset import IGNORE_INDEX, build_supervised_example


def _row(dialogue: str, turn: int, reason: str, *, answer: str) -> dict:
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "help"},
    ]
    if turn > 0:
        messages.extend(
            [
                {"role": "assistant", "content": "which booking?"},
                {"role": "user", "content": "ABC123"},
            ]
        )
    return {
        "messages": messages,
        "answer": {"role": "assistant", "content": answer, "thinking": "private"},
        "metadata": {
            "source_dialog_id": dialogue,
            "turn_index": turn,
            "scenario_id": dialogue.replace("dialog", "scenario"),
            "reason_for_call": reason,
            "correct": 1,
            "reward": 1.0,
        },
    }


def test_loader_keeps_only_final_turn_as_one_complete_dialogue(tmp_path):
    rows = [
        _row("airline_dialog_1", 0, "cancel one flight", answer="old"),
        _row("airline_dialog_1", 1, "cancel one flight", answer="final"),
        _row("airline_dialog_2", 0, "book a flight", answer="done"),
        _row("retail_dialog_1", 0, "return an item", answer="done"),
    ]
    source = tmp_path / "sft.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    dialogues, stats = load_complete_airline_dialogues(source, strict_counts=False)

    assert stats.total_rows == 4
    assert stats.airline_rows == 3
    assert stats.airline_dialogues == 2
    first = next(item for item in dialogues if item.source_dialog_id == "airline_dialog_1")
    assert first.turn_index == 1
    assert first.messages[-1]["content"] == "final"
    assert first.messages[-1]["reasoning"] == "private"


def test_dialogue_level_selection_blocks_near_duplicate_and_never_overlaps(tmp_path):
    rows = [
        _row(f"airline_dialog_{index}", 0, reason, answer="done")
        for index, reason in enumerate(
            [
                "cancel my flight tomorrow",
                "book a new flight to Boston",
                "add two checked bags",
                "change the passenger name",
            ]
        )
    ]
    source = tmp_path / "sft.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    dialogues, _ = load_complete_airline_dialogues(source, strict_counts=False)

    train, validation, audits = select_dialogues(
        dialogues,
        seed=42,
        blocked_reasons=["cancel my flight tomorrow"],
        train_size=2,
        validation_size=1,
    )

    train_ids = {item.source_dialog_id for item in train}
    validation_ids = {item.source_dialog_id for item in validation}
    assert "airline_dialog_0" not in train_ids | validation_ids
    assert train_ids.isdisjoint(validation_ids)
    assert len(audits) == 3

    written = write_dialogue_split(
        train,
        validation,
        audits,
        tmp_path / "out",
        seed=42,
        source_file_hash="hash",
        blocked_reason_count=1,
        similarity_threshold=0.88,
    )
    assert len(written["train"].read_text(encoding="utf-8").splitlines()) == 2
    assert len(written["validation"].read_text(encoding="utf-8").splitlines()) == 1


def test_reason_similarity_detects_exact_normalized_duplicate():
    assert reason_similarity("Cancel flight ABC-123!", "cancel flight abc 123") == 1.0


def test_selection_applies_length_gate_and_one_dialogue_per_intent(tmp_path):
    rows = [
        _row(f"airline_dialog_{index}", 0, reason, answer="done")
        for index, reason in enumerate(
            [
                "cancel my flight tomorrow",
                "Cancel my flight tomorrow!",
                "book a new flight to Boston",
                "add two checked bags",
                "change the passenger name",
                "move my return flight to Friday",
            ]
        )
    ]
    source = tmp_path / "sft.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    dialogues, _ = load_complete_airline_dialogues(source, strict_counts=False)
    token_counts = {dialogue.source_dialog_id: 100 for dialogue in dialogues}
    token_counts["airline_dialog_5"] = 20_000

    first = select_dialogues(
        dialogues,
        seed=42,
        rendered_token_counts=token_counts,
        max_rendered_tokens=16_384,
        max_dialogues_per_intent=1,
        train_size=2,
        validation_size=1,
    )
    second = select_dialogues(
        dialogues,
        seed=42,
        rendered_token_counts=token_counts,
        max_rendered_tokens=16_384,
        max_dialogues_per_intent=1,
        train_size=2,
        validation_size=1,
    )

    selected = first[0] + first[1]
    selected_ids = {dialogue.source_dialog_id for dialogue in selected}
    assert selected_ids == {dialogue.source_dialog_id for dialogue in second[0] + second[1]}
    assert "airline_dialog_5" not in selected_ids
    selected_reasons = [dialogue.reason_for_call for dialogue in selected]
    for index, reason in enumerate(selected_reasons):
        assert all(
            reason_similarity(reason, other) < 0.88
            for other in selected_reasons[index + 1 :]
        )


class _FakeTokenizer:
    role_tokens = {"system": 10, "user": 20, "assistant": 30, "tool": 40}
    content_tokens = {"system": 110, "user": 120, "assistant": 130, "tool": 140}

    def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt):
        assert tokenize is True
        tokens = [1]
        for message in messages:
            role = message["role"]
            tokens.extend([self.role_tokens[role], self.content_tokens[role]])
            if message.get("tool_calls"):
                tokens.append(131)
            tokens.append(99)
        if add_generation_prompt:
            tokens.append(self.role_tokens["assistant"])
        return tokens


def test_assistant_only_mask_labels_content_and_tool_calls_not_observations():
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "help"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "get_user_details", "arguments": {"user_id": "u"}}],
        },
        {"role": "tool", "name": "get_user_details", "content": "result"},
        {"role": "assistant", "content": "done"},
    ]
    example = build_supervised_example(messages, _FakeTokenizer(), tools=[])
    pairs = list(zip(example["input_ids"], example["labels"], strict=True))

    assert any(token == 130 and label == 130 for token, label in pairs)
    assert any(token == 131 and label == 131 for token, label in pairs)
    for observation_token in (110, 120, 140):
        assert all(label == IGNORE_INDEX for token, label in pairs if token == observation_token)
