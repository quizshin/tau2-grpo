"""Regressions for the false-ready 500/150 package discovered on 2026-09-30."""

from copy import deepcopy

import pytest

from tau3_grpo.data.reviewed_sft import (
    REVIEW_AREAS,
    audit_reviewed_package,
    validate_frozen_package,
    validate_rendered_evidence,
)
from tau3_grpo.utils.hashing import sha256_json


def package():
    rows, reviews, tokens = [], [], []
    for split in ("train", "validation"):
        row = {
            "messages": [{"role": "system", "content": "policy"},
                         {"role": "user", "content": split},
                         {"role": "assistant", "content": "Verified result"}],
            "metadata": {"source_dialog_id": split, "source_user_id": split,
                         "split": split, "quality_accepted": True,
                         "curriculum_bucket": "foundation"},
            "supervision": {"version": "approved_assistant_v1", "message_indices": [2]},
        }
        identity = sha256_json([row["messages"], [2]])
        common = {"sample_id": split, "split": split, "messages_mask_sha256": identity}
        reviews.append({**common, "source_user_id": split, "decision": "accepted_codex",
                        "reviewer": "Codex", "issues": [], "evidence": ["source-bound review"],
                        "checks": dict.fromkeys(REVIEW_AREAS, "satisfied"),
                        "native": {"passed": True}})
        tokens.append({**common, "n_total_tokens": 10, "n_label_tokens": 3,
                       "max_length": 100, "native_tokens_unchanged": True,
                       "ignore_nonassistant": True})
        rows.append(row)
    return rows[:1], rows[1:], reviews, tokens


def check(parts):
    return audit_reviewed_package(*parts, sizes=(1, 1))


def test_missing_validation_user_cannot_be_counted_as_zero_overlap():
    parts = package()
    del parts[1][0]["metadata"]["source_user_id"]
    with pytest.raises(ValueError, match="Missing source user"):
        check(parts)


def test_nonempty_supervision_is_not_a_valid_contract():
    parts = package()
    del parts[1][0]["supervision"]["version"]
    with pytest.raises(ValueError, match="approved supervision"):
        check(parts)


def test_declared_acceptance_cannot_bypass_stale_identity():
    parts = package()
    parts[1][0]["messages"][2]["content"] = "Different unreviewed answer"
    with pytest.raises(ValueError, match="Stale evidence"):
        check(parts)


def test_actual_user_overlap_blocks_ready_even_with_all_accept_labels():
    parts = package()
    parts[1][0]["metadata"]["source_user_id"] = "train"
    parts[2][1]["source_user_id"] = "train"
    with pytest.raises(ValueError, match="source users overlap"):
        check(parts)


@pytest.mark.parametrize("kind", ["token_count", "max_length", "semantic_check", "native", "extra_review"])
def test_incomplete_evidence_blocks_freeze(kind):
    parts = package()
    if kind == "token_count":
        parts[3][1]["n_total_tokens"] = None
    elif kind == "max_length":
        parts[3][1]["max_length"] = None
    elif kind == "semantic_check":
        parts[2][1]["checks"]["completion"] = "unknown"
    elif kind == "native":
        parts[2][1]["native"]["passed"] = False
    else:
        parts[2].append({**deepcopy(parts[2][1]), "sample_id": "unused"})
    with pytest.raises(ValueError):
        check(parts)


def test_complete_bound_package_can_be_frozen():
    result = check(package())
    assert result["ready_for_training"] is True
    assert result["validation_unique_users"] == 1
    assert result["user_overlap_count"] == 0


def test_not_ready_manifest_fails_before_any_model_load(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"ready_for_training": false}')
    with pytest.raises(ValueError, match="not ready"):
        validate_frozen_package(manifest, root=tmp_path,
                                train_path=tmp_path / "train", validation_path=tmp_path / "dev")


def test_changed_package_file_fails_before_trusting_review_labels(tmp_path):
    import json

    from tau3_grpo.utils.hashing import sha256_file

    data = tmp_path / "train.jsonl"
    data.write_text("original")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"ready_for_training": True,
                                    "files": {data.name: sha256_file(data)}}))
    data.write_text("tampered")
    with pytest.raises(ValueError, match="Frozen SFT file changed"):
        validate_frozen_package(manifest, root=tmp_path,
                                train_path=data, validation_path=tmp_path / "dev")


@pytest.mark.parametrize("name,size,steps,previous", [
    ("A109", 109, 14, None), ("B393", 393, 50, "A109"), ("C500", 500, 63, "B393"),
])
def test_cumulative_profiles_preserve_update_budget_and_initialization(name, size, steps, previous):
    from pathlib import Path

    import yaml

    from tau3_grpo.training.sft.train import training_schedule

    config = yaml.safe_load(Path(f"configs/train/sft/curriculum_codex_{name}_portable_dev_1epoch.yaml").read_text())
    assert config["data"]["expected_train_size"] == size
    assert config["data"]["expected_validation_size"] == 150
    assert config["data"]["reviewed_package_manifest"]
    assert config["data"]["require_approved_targets"] is True
    assert training_schedule(config["train"], size, 1, 8) == (1, -1, steps)
    assert config["model"].get("requires_previous_stage") == previous
    if previous:
        assert config["model"]["name_or_path"] is None
        assert "SFT_MODEL_NAME_OR_PATH" not in config["launch"]["environment"]


def test_changed_training_tokens_fail_even_if_counts_match():
    from types import SimpleNamespace

    record = {"metadata": {"source_dialog_id": "sample"}}
    example = {"input_ids": [1, 2, 3], "labels": [-100, -100, 3]}
    tokens = [{"sample_id": "sample", "input_ids_sha256": sha256_json([1, 2, 3]),
               "labels_sha256": sha256_json([-100, -100, 3])}]
    dataset = SimpleNamespace(records=[record], examples=[example])
    validate_rendered_evidence(dataset, tokens)
    example["labels"] = [-100, 2, -100]
    with pytest.raises(ValueError, match="loss mask"):
        validate_rendered_evidence(dataset, tokens)


@pytest.fixture
def portable_package(tmp_path):
    import json

    from tau3_grpo.data.compact_sft import export_package
    from tau3_grpo.utils.hashing import sha256_file

    root = tmp_path / "original"
    source = root / "data/old"
    source.mkdir(parents=True)
    archived = root / "results/original_review.txt"
    archived.parent.mkdir()
    archived.write_text("original accepted review and native receipt")
    tools = root / "configs/envs/tool_config.yaml"
    tools.parent.mkdir(parents=True)
    tools.write_text("tools: []\n")
    rows, reviews, tokens = [], [], []
    for split, count in (("train", 500), ("validation", 150)):
        for number in range(count):
            parts = deepcopy(package())
            index = 0 if split == "train" else 1
            row, review, token = parts[index][0], parts[2][index], parts[3][index]
            sid = f"{split}-{number}"
            row["metadata"]["source_dialog_id"] = sid
            row["messages"][1]["content"] = sid
            identity = sha256_json([row["messages"], [2]])
            for item in (review, token):
                item.update(sample_id=sid, messages_mask_sha256=identity)
            review["evidence"] = [{"file": str(archived.relative_to(root)),
                                   "sha256": sha256_file(archived)}]
            rows.append(row)
            reviews.append(review)
            tokens.append(token)

    def jsonl(name, selected):
        (source / name).write_text("".join(json.dumps(row) + "\n" for row in selected))

    jsonl("train.jsonl", rows[:500])
    jsonl("validation.jsonl", rows[500:])
    for stage, count in (("A109", 109), ("B393", 393), ("C500", 500)):
        jsonl(f"train_{stage}.jsonl", rows[:count])
    (source / "review_index.json").write_text(json.dumps(reviews))
    (source / "token_mask_audit.json").write_text(json.dumps(tokens))
    manifest = {
        "schema": "codex_reviewed_sft_package_v1", "ready_for_training": True,
        "train_file": "data/old/train.jsonl", "validation_file": "data/old/validation.jsonl",
        "review_file": "data/old/review_index.json", "token_file": "data/old/token_mask_audit.json",
        "stage_files": {f"data/old/train_{stage}.jsonl": count
                        for stage, count in (("A109", 109), ("B393", 393), ("C500", 500))},
        "audit": audit_reviewed_package(rows[:500], rows[500:], reviews, tokens),
        "files": {str(p.relative_to(root)): sha256_file(p)
                  for p in [*source.iterdir(), tools, archived]},
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "portable"
    receipt = export_package(source / "manifest.json", output, root=root)
    return root, source, output, receipt


def test_portable_package_remains_valid_after_removing_all_history(portable_package, tmp_path):
    import json
    import shutil

    from tau3_grpo.data.reviewed_sft import load_frozen_tokens

    root, source, output, receipt = portable_package
    assert receipt["history_required_at_training"] is False
    assert (output / "train.jsonl").read_bytes() == (source / "train.jsonl").read_bytes()
    assert (output / "validation.jsonl").read_bytes() == (source / "validation.jsonl").read_bytes()
    assert not (output / "train_C500.jsonl").exists()  # byte-identical duplicate removed
    relocated = tmp_path / "isolated/assets/sft"
    shutil.copytree(output, relocated)
    shutil.rmtree(root)
    shutil.rmtree(output)
    manifest = json.loads((relocated / "manifest.json").read_text())
    for stage in manifest["stage_files"]:
        checked = validate_frozen_package(
            relocated / "manifest.json", root=tmp_path / "isolated",
            train_path=relocated / stage, validation_path=relocated / "validation.jsonl",
        )
        assert len(load_frozen_tokens(relocated / "manifest.json", checked,
                                      root=tmp_path / "isolated")) == 650


@pytest.mark.parametrize("name", ["train.jsonl", "review_index.json", "token_mask_audit.json"])
def test_portable_artifact_cannot_be_replaced_by_merely_rehashing_it(portable_package, name):
    import json

    from tau3_grpo.utils.hashing import sha256_file

    _, _, output, _ = portable_package
    path = output / name
    path.write_text(path.read_text() + "\n")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][name] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs from its frozen source"):
        validate_frozen_package(manifest_path, root=output,
                                train_path=output / "train.jsonl",
                                validation_path=output / "validation.jsonl")


def test_portable_missing_protected_artifact_is_rejected(portable_package):
    import json

    _, _, output, _ = portable_package
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["files"]["review_index.json"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Missing protected"):
        validate_frozen_package(manifest_path, root=output,
                                train_path=output / "train.jsonl",
                                validation_path=output / "validation.jsonl")


def test_portable_auxiliary_binding_cannot_override_token_identity(portable_package):
    import json

    from tau3_grpo.data.reviewed_sft import _portable_source_binding

    _, _, output, _ = portable_package
    manifest = json.loads((output / "manifest.json").read_text())
    source = json.loads((output / "source_manifest.json").read_text())
    manifest["source_package"]["artifact_bindings"]["token_mask_audit.json"] = "configs/envs/tool_config.yaml"
    manifest["files"]["token_mask_audit.json"] = source["files"]["configs/envs/tool_config.yaml"]
    reviews = json.loads((output / "review_index.json").read_text())
    with pytest.raises(ValueError, match="differs from its frozen source"):
        _portable_source_binding(manifest, output, reviews)


@pytest.mark.parametrize("name", ["../outside.json", "/outside.json"])
def test_portable_paths_cannot_escape_package(portable_package, name):
    import json

    _, _, output, _ = portable_package
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = {name: "fake"}
    manifest["files"].update({key: "fake" for key in [
        manifest["train_file"], manifest["validation_file"], manifest["review_file"],
        manifest["token_file"], manifest["tool_config_file"],
        manifest["source_package"]["manifest_file"], *manifest["stage_files"],
    ]})
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inside the package"):
        validate_frozen_package(manifest_path, root=output,
                                train_path=output / "train.jsonl",
                                validation_path=output / "validation.jsonl")


def test_portable_export_never_overwrites_a_frozen_package(portable_package):
    from tau3_grpo.data.compact_sft import export_package

    root, source, output, _ = portable_package
    with pytest.raises(FileExistsError):
        export_package(source / "manifest.json", output, root=root)


@pytest.mark.parametrize("stage", ["A109", "B393", "C500"])
def test_portable_profiles_change_storage_only(stage):
    from pathlib import Path

    import yaml

    from tau3_grpo.launch import prepare

    folder = Path("configs/train/sft")
    import json

    old = json.loads(Path("tests/fixtures/reviewed_sft_balanced_recipe.json").read_text())[stage]
    path = folder / f"curriculum_codex_{stage}_portable_dev_1epoch.yaml"
    new = yaml.safe_load(path.read_text())
    for key in ("model", "train", "lora", "includes", "launch"):
        assert new[key] == old[key]
    for key in old["data"].keys() - {
        "train_jsonl", "validation_jsonl", "tool_config", "reviewed_package_manifest",
    }:
        assert new["data"][key] == old["data"][key]
    assert "portable_codex_20261001" in new["data"]["reviewed_package_manifest"]
    command, _, snapshot = prepare("sft", path, "e0", 42, [], {})
    assert command and snapshot["configuration"]["data"] == new["data"]
