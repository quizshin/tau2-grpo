"""The RL simulator must receive the same complete scenario as native tau2."""

import json

import pytest

from tau3_grpo.data.schema import ArealTaskRecord
from tau3_grpo.envs.adapter import adapt_record, load_flight_db
from tau3_grpo.envs.tau2_bridge import user_simulator_cls


@pytest.mark.tau3
@pytest.mark.parametrize("task_id,user_id", [
    ("airline_709", "emma_kim_4489"),
    ("airline_871", "victoria_lewis_3bf152"),
    ("airline_382", "riley_wilson_9ad6e5"),
])
def test_real_rl_tasks_deliver_known_user_ids_to_simulator(requires_tau2, areal_jsonl, task_id, user_id):
    records = (json.loads(line) for line in areal_jsonl.read_text().splitlines())
    record = ArealTaskRecord.model_validate(next(row for row in records if row["id"] == task_id))
    adapted = adapt_record(record, dataset_root=areal_jsonl.parent)
    assert user_id in load_flight_db(adapted.db_path).users
    simulator = user_simulator_cls()(llm="openai/test-no-network", instructions=adapted.user_instructions)
    assert user_id in simulator.system_prompt
    assert record.user_scenario["instructions"]["known_info"] in simulator.system_prompt
    assert adapted.user_instructions == str(adapted.task.user_scenario)


@pytest.mark.tau3
def test_scenario_preserves_persona_known_and_unknown_info(requires_tau2, tmp_path):
    (tmp_path / "db.json").write_text("{}")
    record = ArealTaskRecord(id="scenario", db_path="db.json", evaluation_criteria={},
        user_scenario={"persona": "A cautious passenger", "instructions": {
            "domain": "airline", "reason_for_call": "Change a flight", "known_info": "user_id: example_42",
            "unknown_info": "Does not know the change fee", "task_instructions": "Ask about options"}})
    adapted = adapt_record(record, dataset_root=tmp_path)
    for text in ("A cautious passenger", "Change a flight", "example_42", "Does not know the change fee", "Ask about options"):
        assert text in adapted.user_instructions
