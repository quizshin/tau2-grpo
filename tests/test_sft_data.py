"""AReaL complete-dialogue selection and assistant-only SFT masking."""

from __future__ import annotations

import json

from tau3_grpo.data.sft import (
    load_complete_airline_dialogues,
    reason_similarity,
    select_dialogues,
    write_dialogue_split,
)
from tau3_grpo.training.sft.dataset import IGNORE_INDEX, build_supervised_example


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



def test_staged_screen_matches_legacy_similarity():
    from tau3_grpo.data.staged_sft import near_duplicate
    reasons = ['Cancel my flight tomorrow', 'cancel my flight tomorrow!',
               'book a new flight to Boston', 'add a bag', '', 'add two bags']
    for a in reasons:
        for b in reasons:
            assert near_duplicate(a, b) == (reason_similarity(a, b) >= 0.88)


def test_ordered_receipts_reject_swapped_same_tool_ids():
    from tau3_grpo.data.staged_sft import ordered_tool_receipts
    row = {'messages': [
        {'role': 'assistant', 'tool_calls': [
            {'id': 'a', 'function': {'name': 'get_user_details'}},
            {'id': 'b', 'function': {'name': 'get_user_details'}}]},
        {'role': 'tool', 'name': 'get_user_details', 'tool_call_id': 'a'},
        {'role': 'tool', 'name': 'get_user_details', 'tool_call_id': 'b'}]}
    assert ordered_tool_receipts(row) == []
    row['messages'][1]['tool_call_id'] = 'b'
    assert '1:tool_id_mismatch' in ordered_tool_receipts(row)



def test_baggage_rules_use_prior_membership_and_passenger_count():
    from tau3_grpo.data.sft_policy_checks import audit_baggage_allowances
    from tau3_grpo.prompts import build_system_prompt
    messages = [
        {'role': 'system', 'content': build_system_prompt()},
        {'role': 'tool', 'name': 'get_user_details',
         'content': json.dumps({'user_id': 'u', 'membership': 'silver'})},
        {'role': 'tool', 'name': 'get_reservation_details', 'content': json.dumps({
            'reservation_id': 'r', 'user_id': 'u', 'cabin': 'economy',
            'passengers': [{}, {}], 'total_baggages': 3, 'nonfree_baggages': 0})},
        {'role': 'assistant', 'tool_calls': [{'name': 'update_reservation_baggages',
            'arguments': {'reservation_id': 'r', 'total_baggages': 4, 'nonfree_baggages': 0}}]}]
    check = audit_baggage_allowances(messages)[0]
    assert check['status'] == 'satisfied' and check['free_total'] == 4
    messages[-1]['tool_calls'][0]['arguments']['nonfree_baggages'] = 1
    assert audit_baggage_allowances(messages)[0]['status'] == 'violated'
    # Future profile evidence cannot retroactively establish a prerequisite.
    messages.append(messages.pop(1))
    assert audit_baggage_allowances(messages)[0]['status'] == 'unknown'



def test_review_acceptance_requires_bound_visible_messages(tmp_path):
    import pytest

    from tau3_grpo.analysis.rubric_pilot import visible_events
    from tau3_grpo.data.finalize_staged_sft import reviewed_pool
    from tau3_grpo.prompts import build_system_prompt
    from tau3_grpo.utils.hashing import sha256_json
    row = {'messages': [{'role': 'system', 'content': build_system_prompt()},
                        {'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hello'}],
           'metadata': {'source_dialog_id': 'test'},
           'supervision': {'basis': 'source_answer_positions'}}
    (tmp_path / 'records').mkdir()
    path = tmp_path / 'records/test.json'
    verdict = {'visible_hash': sha256_json(visible_events(row['messages'])), 'status': 'review_ready',
               'audit': {'recommendation': 'keep_candidate', 'issues': [],
                         'requirements': [{'status': 'satisfied'}], 'difficulty': {'level': 'easy'}}}
    path.write_text(json.dumps(verdict))
    accepted, held = reviewed_pool([row], tmp_path)
    assert len(accepted) == 1 and not held
    verdict['audit']['requirements'][0]['status'] = 'unknown'
    path.write_text(json.dumps(verdict))
    assert not reviewed_pool([row], tmp_path)[0]
    row['messages'][-1]['content'] = 'Changed after judgment'
    with pytest.raises(ValueError, match='identity mismatch'):
        reviewed_pool([row], tmp_path)



def test_continuation_rejects_partial_epoch_and_changed_training_masks(tmp_path):
    import hashlib
    import pytest
    from tau3_grpo.training.sft.continuation import validate_training
    root = tmp_path
    (root / 'adapter').mkdir()
    (root / 'train.exit').write_text('0')
    stats = {'train': {'dialogues': 100}, 'validation': {'dialogues': 52}}
    summary = {'actual_optimizer_steps': 13, 'train_loss': 1.2,
               'validation_metrics': {'eval_loss': 1.0}, **stats}
    (root / 'adapter/train_summary.json').write_text(json.dumps(summary))
    state = {'global_step': 13, 'epoch': 1, 'log_history': [{'grad_norm': 1.0}]}
    path = root / 'adapter/trainer_state.json'
    path.write_text(json.dumps(state))
    files = {}
    for split in ('train', 'validation'):
        (root / 'adapter' / f'effective_{split}.jsonl').write_text(split)
        files[f'{split}_effective.jsonl'] = hashlib.sha256(split.encode()).hexdigest()
    audit = {'token_stats': stats, 'files': files}
    assert validate_training(root, audit)['actual_optimizer_steps'] == 13
    with pytest.raises(ValueError, match='complete'):
        validate_training(root, audit, expected_steps=39, expected_epochs=3)
    state.update(global_step=39, epoch=3)
    summary['actual_optimizer_steps'] = 39
    path.write_text(json.dumps(state))
    (root / 'adapter/train_summary.json').write_text(json.dumps(summary))
    assert validate_training(root, audit, expected_steps=39, expected_epochs=3)
    state.update(global_step=13, epoch=1)
    summary['actual_optimizer_steps'] = 13
    (root / 'adapter/train_summary.json').write_text(json.dumps(summary))
    state['epoch'] = .92
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='complete'):
        validate_training(root, audit)
    state['epoch'] = 1
    path.write_text(json.dumps(state))
    (root / 'adapter/effective_train.jsonl').write_text('changed mask')
    with pytest.raises(ValueError, match='masks/messages'):
        validate_training(root, audit)


def test_continuation_tokenizer_rejects_semantic_change(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    import transformers
    from tau3_grpo.training.sft.continuation import preserve_base_tokenizer

    def tokenizer(path, **kwargs):
        return SimpleNamespace(get_vocab=lambda: {'token': 1 if path.endswith('base') else 2})

    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', tokenizer)
    with pytest.raises(ValueError, match='semantics changed'):
        preserve_base_tokenizer(tmp_path / 'base', tmp_path / 'merged', tmp_path / 'backup')
    assert not (tmp_path / 'backup').exists()
