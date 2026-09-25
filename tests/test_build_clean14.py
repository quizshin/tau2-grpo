"""Whole-dataset clean14 validation, not just new-candidate filtering."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tau3_grpo.data import build_clean14 as builder
from tau3_grpo.training.sft.train import training_schedule
from tau3_grpo.utils.hashing import sha256_text


def example(sid="airline_dialog_1", source="areal_tau2_airline_sft"):
    return {"messages": [
        {"role": "system", "content": "policy"}, {"role": "user", "content": "help"},
        {"role": "assistant", "tool_calls": [{"name": "lookup", "arguments": {"id": "u"}}]},
        {"role": "tool", "name": "lookup", "content": "response"},
        {"role": "assistant", "content": "done"}],
        "metadata": {"source": source, "source_dialog_id": sid, "reason_for_call": "find a booking"}}


@pytest.fixture
def context(monkeypatch):
    monkeypatch.setattr(builder, "prepare_agent_messages", lambda messages: messages)
    monkeypatch.setattr(builder, "build_supervised_example", lambda *a, **kw: {
        "input_ids": [1, 2, 3], "labels": [-100, 2, 3], "n_total_tokens": 3, "n_label_tokens": 2})
    row = example()
    return dict(role="train", good={builder._sid(row): True},
        source_records={builder._sid(row): deepcopy(row)}, entries={}, blocked=[],
        schemas=[{"function": {"name": "lookup", "parameters": {"type": "object",
            "properties": {"id": {"type": "string"}}, "required": ["id"]}}}],
        tokenizer=None, max_length=100, similarity_threshold=.88)


@pytest.mark.parametrize("role", ["train", "validation"])
def test_all_source_rows_need_explicit_quality_and_original_messages(context, role):
    context["role"] = role
    assert builder.validate_record(example(), **context)["label_tokens"] == 2
    for good in [{}, {"airline_dialog_1": False}]:
        with pytest.raises(ValueError, match="source quality"):
            builder.validate_record(example(), **{**context, "good": good})
    changed = example()
    changed["messages"][-1]["content"] = "tampered"
    with pytest.raises(ValueError, match="messages were changed"):
        builder.validate_record(changed, **context)


def test_anchors_and_supplemental_cannot_leak_or_duplicate():
    with pytest.raises(ValueError, match="overlap"):
        builder.validate_identities([example()], [example()])
    with pytest.raises(ValueError, match="Duplicate"):
        builder.validate_identities([example(), example()], [])
    with pytest.raises(ValueError, match="Known bad"):
        builder.validate_identities([example("airline_dialog_837")], [])
    with pytest.raises(ValueError, match="Missing"):
        builder.validate_identities([{"metadata": {}}], [])


def test_every_row_gets_schema_overlap_and_mask_checks(context, monkeypatch):
    row = example()
    row["messages"][2]["tool_calls"][0]["arguments"] = {}
    context["source_records"][builder._sid(row)] = deepcopy(row)
    with pytest.raises(ValueError, match="schema"):
        builder.validate_record(row, **context)
    context["source_records"][builder._sid(row)] = example()
    with pytest.raises(ValueError, match="heldout"):
        builder.validate_record(example(), **{**context, "blocked": ["find a booking"]})
    for field, value in [("labels", [-100, 7, 3]), ("n_label_tokens", 5), ("n_total_tokens", 1)]:
        render = {"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "n_total_tokens": 3, "n_label_tokens": 2}
        render[field] = value
        monkeypatch.setattr(builder, "build_supervised_example", lambda *a, r=render, **kw: r)
        with pytest.raises(ValueError, match="mask"):
            builder.validate_record(example(), **context)


def test_invalid_dialogues_and_missing_intent_rejected(context):
    for messages in [[], [{"role": "assistant", "content": "done"}]]:
        row = example()
        row["messages"] = messages
        with pytest.raises(ValueError, match="complete"):
            builder.validate_record(row, **context)
    row = example()
    row["metadata"]["reason_for_call"] = ""
    with pytest.raises(ValueError, match="intent"):
        builder.validate_record(row, **context)


def supplemental():
    row = example("authored_task", "areal_rl_train_authored_executed")
    row["metadata"].update(source_task_id="task", source_task_hash="taskhash", source_db_hash="dbhash",
        source_revision="revision", verification="tool_execution_and_reference_final_db_equality",
        independent_task_success_evaluation=False, final_db_hash="a" * 64,
        dialogue_hash=sha256_text(json.dumps(row["messages"], sort_keys=True)),
        executions=[{"name": "lookup", "arguments": {"id": "u"}, "response_sha256": sha256_text("response")}])
    entries = {"task": {"task_id": "task", "split": "train", "task_hash": "taskhash",
                        "db_hash": "dbhash", "source_revision": "revision"}}
    return row, entries


def test_supplemental_needs_real_bound_execution_receipts(context):
    row, entries = supplemental()
    assert builder.validate_record(row, **{**context, "entries": entries})["label_tokens"] == 2
    for field, value in [("verification", "claimed"), ("executions", []), ("source_task_hash", "other"),
                         ("dialogue_hash", "wrong"), ("independent_task_success_evaluation", True)]:
        changed = deepcopy(row)
        changed["metadata"][field] = value
        with pytest.raises(ValueError):
            builder.validate_record(changed, **{**context, "entries": entries})
    entries["task"]["split"] = "selection"
    with pytest.raises(ValueError, match="training manifest"):
        builder.validate_supplemental(row, entries)


def test_supplemental_response_receipt_tamper_is_rejected():
    row, entries = supplemental()
    row["metadata"]["executions"][0]["response_sha256"] = "bad"
    with pytest.raises(ValueError, match="receipt"):
        builder.validate_supplemental(row, entries)


def test_source_quality_fails_closed_for_missing_or_failed_turn(tmp_path):
    path = tmp_path / "source.jsonl"
    path.write_text('\n'.join(json.dumps({"metadata": {"source_dialog_id": "airline_dialog_1",
                    "correct": c, "reward": 1, "seed_pattern_task_id": "scenario_pos_pos"}}) for c in [1, 0, 1]))
    good, _ = builder._source_quality(path)
    assert not good["airline_dialog_1"] and not good.get("missing", False)


def test_frozen_recipes_have_separate_counts_and_hashes():
    root = Path(__file__).resolve().parents[1]
    for size in (96, 100):
        recipe = builder.load_recipe(root / f"configs/data/clean14_{size}_frozen_v1.json")
        assert recipe["train_count"] == size
        assert len(recipe["selected_areal_ids"]) + 3 == size
        assert sum(recipe["additional_category_counts"].values()) == size - 44
        assert len(recipe["outputs_sha256"]["clean14_v1_train.jsonl"]) == 64


@pytest.mark.parametrize("size,config_name,steps", [
    (96, "clean14_v1_lora_1epoch.yaml", 12), (100, "clean14_v1_100_lora_1epoch.yaml", 13),
])
def test_registered_sft_configs_match_recipe_and_training_budget(size, config_name, steps):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/train/sft" / config_name).read_text())
    assert config["data"]["expected_train_size"] == size
    assert config["data"]["expected_validation_size"] == 5
    assert config["data"]["max_length"] == builder.load_recipe(root / f"configs/data/clean14_{size}_frozen_v1.json")["max_length"]
    train = config["train"]
    assert training_schedule(train, size, train["per_device_batch_size"], train["gradient_accumulation_steps"]) == (1, -1, steps)
    assert (root / config["launch"]["entrypoint"]).is_file()
    assert "/autodl-fs/" in config["output"]["dir"]


def test_bad_recipe_counts_fail_and_existing_output_is_never_touched(tmp_path):
    root = Path(__file__).resolve().parents[1]
    recipe = json.loads((root / "configs/data/clean14_96_frozen_v1.json").read_text())
    recipe["train_count"] = 100
    p = tmp_path / "recipe.json"
    p.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="count mismatch"):
        builder.load_recipe(p)
    out = tmp_path / "existing"
    out.mkdir()
    (out / "keep").write_text("history")
    with pytest.raises(FileExistsError):
        builder.build(p, {}, tmp_path, out)
    assert (out / "keep").read_text() == "history"


def test_input_tampering_rejected_before_tokenizer_load(tmp_path):
    paths = {n: tmp_path / n for n in builder.INPUT_NAMES}
    for p in paths.values():
        p.write_text("input")
    recipe = {"inputs_sha256": {n: builder.sha256_file(p) for n, p in paths.items()}}
    paths["supplemental"].write_text("tampered")
    with pytest.raises(ValueError, match="identity changed"):
        builder.verify_inputs(recipe, paths, tmp_path)
