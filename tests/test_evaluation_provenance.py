import json

import pytest

from tau3_grpo.data.schema import ArealTaskRecord
from tau3_grpo.envs.adapter import adapt_record
from tau3_grpo.evaluation.provenance import evaluation_provenance


def test_actual_tasks_databases_and_sources_are_attested(areal_jsonl, tmp_path):
    record = ArealTaskRecord.model_validate(json.loads(areal_jsonl.read_text().splitlines()[0]))
    task = adapt_record(record, dataset_root=areal_jsonl.parent)
    db = tmp_path / "db.json"
    db.write_bytes(task.db_path.read_bytes())
    jobs = [{"task_id": task.task_id, "task": task.task, "db_path": db}]
    first = evaluation_provenance(jobs)
    assert first == evaluation_provenance(jobs * 4)
    assert first["evaluator_source_sha256"]
    assert "tau3_grpo/envs/session.py" in first["harness_source_sha256"]
    db.write_bytes(db.read_bytes() + b"\n")
    second = evaluation_provenance(jobs)
    assert first["task_db_sha256"] != second["task_db_sha256"]
    assert first["task_manifest_sha256"] == second["task_manifest_sha256"]
    altered = task.task.model_copy(update={"id": "different"})
    with pytest.raises(ValueError, match="different task definitions"):
        evaluation_provenance(jobs + [{**jobs[0], "task": altered}])
