"""Export an accepted v1 SFT package into a portable, local-file-only v2 package.

The source's full archive guard runs once at export. Training thereafter verifies
the exact final conversations, accepted review ledger and token/mask inventory.
Historical source identities remain recorded, but their paths are not opened.
No generation, semantic acceptance, model download or GPU operation occurs here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from tau3_grpo.data.reviewed_sft import PORTABLE_SCHEMA, validate_frozen_package
from tau3_grpo.paths import CODE_ROOT
from tau3_grpo.utils.hashing import sha256_file


def export_package(source_manifest, output, *, root):
    """Freeze a new package; never overwrite the source or an existing output."""
    root, source_manifest, output = map(lambda p: Path(p).resolve(), (root, source_manifest, output))
    if output.exists():
        raise FileExistsError(f"Package already exists: {output}")
    previous = json.loads(source_manifest.read_text())
    if previous.get("schema") != "codex_reviewed_sft_package_v1":
        raise ValueError("Portable export requires an accepted v1 source package")
    source_sha = sha256_file(source_manifest)
    full_stage = next((name for name, size in previous["stage_files"].items()
                       if size == previous["audit"]["train_count"]
                       and previous["files"][name] == previous["files"][previous["train_file"]]), None)
    if full_stage is None:
        raise ValueError("Source has no exact full-training cumulative stage")
    validate_frozen_package(
        source_manifest, root=root, train_path=root / full_stage,
        validation_path=root / previous["validation_file"],
    )
    source_dir = source_manifest.parent
    bindings = {}
    names = {}
    for field in ("train_file", "validation_file", "review_file", "token_file"):
        original = previous[field]
        names[field] = Path(original).name
        bindings[names[field]] = original
    stages, stage_bindings = {}, {}
    for original, size in previous["stage_files"].items():
        name = Path(original).name
        if previous["files"][original] == previous["files"][previous["train_file"]]:
            name = names["train_file"]
        if name in stages:
            raise ValueError("Ambiguous cumulative stage filenames")
        stages[name], stage_bindings[name] = size, original
        bindings[name] = original
    # Prompt/tool snapshots are small runtime inputs. Tokenizer identity is kept
    # as metadata; the real tokenizer is checked against exact rendered IDs later.
    tool_source = "configs/envs/tool_config.yaml"
    if tool_source not in previous["files"]:
        raise ValueError("Source package has no frozen tool config identity")
    bindings["tool_config.yaml"] = tool_source
    for name in ("frozen_prompt_protocols.json", "token_summary.json", "stage_token_summary.json",
                 "selection.json", "rejected_index.json"):
        original = str((source_dir / name).relative_to(root))
        if original in previous["files"]:
            bindings[name] = original
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".compact-sft-", dir=output.parent) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        for name, original in bindings.items():
            shutil.copyfile(root / original, staging / name)
        shutil.copyfile(source_manifest, staging / "source_manifest.json")
        if sha256_file(staging / "source_manifest.json") != source_sha:
            raise ValueError("Source manifest changed during export")
        manifest = {
            "schema": PORTABLE_SCHEMA, "path_scope": "package",
            "ready_for_training": True, "status": "frozen_cpu_verified_gpu_not_started",
            **names, "stage_files": stages, "tool_config_file": "tool_config.yaml",
            "source_package": {
                "manifest_file": "source_manifest.json", "manifest_sha256": source_sha,
                "original_path": str(source_manifest.relative_to(root)),
                "artifact_bindings": bindings, "stage_bindings": stage_bindings,
                "verification": "full_v1_files_reviews_and_selected_stage_verified_at_export",
                "historical_paths": "provenance_only_not_training_dependencies",
            },
            **{key: previous[key] for key in (
                "audit", "tokenizer_files", "tokenizer_source", "tokenizer_chat_template_sha256",
                "token_summary", "stage_token_summary", "semantic_review_modes",
                "validation_unique_goal_count", "replaced_train_count",
            ) if key in previous},
            "gpu_training_started": False,
            "files": {p.name: sha256_file(p) for p in sorted(staging.iterdir())},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        for name in stages:
            validate_frozen_package(
                staging / "manifest.json", root=staging,
                train_path=staging / name, validation_path=staging / names["validation_file"],
            )
        staging.rename(output)
    return {
        "schema": manifest["schema"], "package": str(output),
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "parent_manifest_sha256": source_sha, "audit": manifest["audit"],
        "protected_files": len(manifest["files"]),
        "previous_protected_files": len(previous["files"]),
        "bytes": sum(p.stat().st_size for p in output.iterdir()),
        "history_required_at_training": False, "gpu_started": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=CODE_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(export_package(args.source_manifest, args.output, root=args.root), indent=2))


if __name__ == "__main__":
    main()
