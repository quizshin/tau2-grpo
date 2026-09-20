"""Capture inputs and actual source identity before a new evaluation starts."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

from tau3_grpo.envs.adapter import load_default_flight_db
from tau3_grpo.paths import CODE_ROOT


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_hashes(root, paths):
    files = {}
    for name in paths:
        path = root / name
        candidates = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for candidate in candidates:
            files[str(candidate.relative_to(root))] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    return files


def evaluation_provenance(jobs):
    tasks, databases = {}, {}
    file_cache = {}
    default_db_hash = None
    for job in jobs:
        task = job["task"].model_dump(mode="json")
        task_hash = digest(task)
        task_id = job["task_id"]
        if task_id in tasks and tasks[task_id] != task_hash:
            raise ValueError("A task ID refers to different task definitions")
        tasks[task_id] = task_hash
        path = job["db_path"]
        if path is None:
            if default_db_hash is None:
                default_db_hash = digest(load_default_flight_db().model_dump(mode="json"))
            database = {"representation": "canonical_model_json", "sha256": default_db_hash}
        else:
            path = Path(path).resolve()
            if path not in file_cache:
                file_cache[path] = hashlib.sha256(path.read_bytes()).hexdigest()
            database = {"representation": "source_file_bytes", "sha256": file_cache[path]}
        if task_id in databases and databases[task_id] != database:
            raise ValueError("A task ID refers to different initial databases")
        databases[task_id] = database
    # Use the installed benchmark source rather than assuming the checkout is imported.
    benchmark = Path(importlib.import_module("tau2").__file__).parent
    evaluators = source_hashes(benchmark, ["evaluator"])
    harness = source_hashes(CODE_ROOT, ["tau3_grpo/envs", "tau3_grpo/evaluation", "tau3_grpo/prompts.py"])
    harness.update({"tau2/" + name: value for name, value in source_hashes(benchmark, [
        "orchestrator", "runner", "agent", "user", "environment", "data_model",
        "domains/airline",
    ]).items()})
    return {
        "provenance_schema": "tau3_evaluation_inputs_v1",
        "task_manifest_sha256": digest(tasks), "task_definition_sha256": tasks,
        "task_manifest_representation": "canonical_task_id_to_definition_sha256_map",
        "task_db_sha256": databases,
        "evaluator_source_sha256": evaluators, "harness_source_sha256": harness,
        "provenance_scope": "pre_run_inputs_and_source_files_not_live_service_weights",
    }
