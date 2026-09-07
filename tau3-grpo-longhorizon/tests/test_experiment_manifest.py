"""Experiment manifest, checkpoint resume, and telemetry aggregation."""

from __future__ import annotations

import json

import pytest

from tau3_grpo.experiment.manifest import (
    build_schedule,
    create_manifest,
    flatten_schedule,
    read_manifest,
    resume_from,
    verify_extension,
    verify_resume,
    write_manifest,
)
from tau3_grpo.experiment.telemetry import (
    TelemetryWriter,
    build_report,
    build_update_record,
    summarise_telemetry,
    write_report,
)

TASKS = [f"airline_{i}" for i in range(20)]


def _manifest(**overrides):
    kwargs = {
        "experiment": "e3",
        "arm": "e3",
        "seed": 42,
        "task_ids": TASKS,
        "train_split_hash": "split-hash",
        "group_size": 8,
        "groups_per_update": 16,
        "total_updates": 4,
    }
    kwargs.update(overrides)
    return create_manifest(**kwargs)


def test_schedule_is_deterministic():
    a = build_schedule(TASKS, seed=42, group_size=8, groups_per_update=4, total_updates=3)
    b = build_schedule(TASKS, seed=42, group_size=8, groups_per_update=4, total_updates=3)
    assert [item.task_ids for item in a] == [item.task_ids for item in b]


def test_schedule_changes_with_seed():
    a = build_schedule(TASKS, seed=42, group_size=8, groups_per_update=4, total_updates=3)
    b = build_schedule(TASKS, seed=43, group_size=8, groups_per_update=4, total_updates=3)
    assert [item.task_ids for item in a] != [item.task_ids for item in b]


def test_schedule_has_expected_shape():
    schedule = build_schedule(
        TASKS, seed=42, group_size=8, groups_per_update=16, total_updates=5
    )
    assert len(schedule) == 5
    assert all(len(item.task_ids) == 16 for item in schedule)
    assert [item.update_index for item in schedule] == [0, 1, 2, 3, 4]


def test_flatten_schedule_is_exact_update_order():
    schedule = build_schedule(
        TASKS, seed=42, group_size=8, groups_per_update=3, total_updates=2
    )
    assert flatten_schedule(schedule) == [
        *schedule[0].task_ids,
        *schedule[1].task_ids,
    ]


def test_schedule_cycles_when_pool_is_small():
    schedule = build_schedule(
        ["a", "b"], seed=42, group_size=8, groups_per_update=4, total_updates=1
    )
    assert sorted(schedule[0].task_ids) == ["a", "a", "b", "b"]


def test_schedule_rejects_empty_pool():
    with pytest.raises(ValueError, match="empty task pool"):
        build_schedule([], seed=42, group_size=8, groups_per_update=1, total_updates=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_size": 0},
        {"groups_per_update": 0},
        {"total_updates": 0},
    ],
)
def test_schedule_rejects_nonpositive(kwargs):
    base = {"seed": 42, "group_size": 8, "groups_per_update": 4, "total_updates": 2}
    base.update(kwargs)
    with pytest.raises(ValueError, match="must be positive"):
        build_schedule(TASKS, **base)


def test_manifest_roundtrip(tmp_path):
    manifest = _manifest()
    write_manifest(manifest, tmp_path)
    reloaded = read_manifest(tmp_path)
    assert reloaded.schedule_hash == manifest.schedule_hash
    assert reloaded.seed == 42
    assert reloaded.group_size == 8
    assert reloaded.groups_per_update == 16
    assert len(reloaded.schedule) == len(manifest.schedule)


def test_manifest_reader_rejects_tampered_schedule(tmp_path):
    manifest = _manifest()
    path = write_manifest(manifest, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schedule"][0]["task_ids"][0] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schedule_hash does not match"):
        read_manifest(tmp_path)


def test_schedule_hash_is_stable():
    assert _manifest().schedule_hash == _manifest().schedule_hash


def test_schedule_hash_tracks_split_hash():
    assert _manifest().schedule_hash != _manifest(train_split_hash="other").schedule_hash


def test_resume_returns_tail():
    manifest = _manifest(total_updates=5)
    tail = resume_from(manifest, 2)
    assert len(tail) == 3
    assert tail[0].update_index == 2


def test_resume_from_zero_is_full_schedule():
    manifest = _manifest(total_updates=5)
    assert len(resume_from(manifest, 0)) == 5


def test_resume_at_end_is_empty():
    manifest = _manifest(total_updates=5)
    assert resume_from(manifest, 5) == []


def test_resume_rejects_out_of_range():
    manifest = _manifest(total_updates=3)
    with pytest.raises(ValueError, match="exceeds scheduled"):
        resume_from(manifest, 4)
    with pytest.raises(ValueError, match="cannot be negative"):
        resume_from(manifest, -1)


def test_verify_resume_accepts_identical():
    verify_resume(_manifest(), _manifest())


def test_verify_resume_detects_schedule_drift():
    with pytest.raises(ValueError, match="schedule hash mismatch"):
        verify_resume(_manifest(seed=42), _manifest(seed=43))


def test_verify_extension_accepts_40_to_60_prefix():
    original = _manifest(total_updates=40)
    extended = _manifest(total_updates=60)
    verify_extension(original, extended)


def test_verify_extension_rejects_truncation_or_seed_drift():
    with pytest.raises(ValueError, match="must increase"):
        verify_extension(_manifest(total_updates=60), _manifest(total_updates=40))
    with pytest.raises(ValueError, match="changes seed"):
        verify_extension(_manifest(total_updates=40), _manifest(total_updates=60, seed=43))


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no experiment manifest"):
        read_manifest(tmp_path)


# ---- telemetry ---------------------------------------------------------


def test_update_record_captures_stats():
    from tau3_grpo.algo.dynamic_filtering import evaluate_groups

    stats = evaluate_groups([0.0] * 8, ["u"] * 8, group_size=8)
    record = build_update_record(
        update_index=0,
        arm="e1",
        seed=42,
        rewards=[0.0] * 8,
        filter_stats=stats,
        failure_categories=["turn_limit", "turn_limit", "wrong_outcome"],
        termination_reasons=["max_steps", "max_steps", "agent_stop"],
    )
    assert record.mean_reward == 0.0
    assert record.dynamic_filter["d_bar"] == 1.0
    assert record.failure_categories["turn_limit"] == 2
    assert record.termination_reasons["max_steps"] == 2


def test_telemetry_writer_appends_jsonl(tmp_path):
    writer = TelemetryWriter(tmp_path)
    for index in range(3):
        writer.append(
            build_update_record(
                update_index=index, arm="e1", seed=42, rewards=[float(index % 2)]
            )
        )
    records = writer.read_all()
    assert len(records) == 3
    assert [r["update_index"] for r in records] == [0, 1, 2]


def test_telemetry_read_all_on_missing_file(tmp_path):
    assert TelemetryWriter(tmp_path / "nope").read_all() == []


def test_summarise_separates_failure_from_unfinished():
    records = [
        build_update_record(
            update_index=0,
            arm="e1",
            seed=42,
            rewards=[0.0],
            failure_categories=["turn_limit", "wrong_outcome"],
        ).to_dict()
    ]
    summary = summarise_telemetry(records)
    assert summary["failure_categories"]["turn_limit"] == 1
    assert summary["failure_categories"]["wrong_outcome"] == 1


def test_summarise_empty():
    assert summarise_telemetry([]) == {"updates": 0}


def test_summarise_tracks_reward_trajectory():
    records = [
        build_update_record(update_index=i, arm="e1", seed=42, rewards=[float(i) / 10]).to_dict()
        for i in range(4)
    ]
    summary = summarise_telemetry(records)
    assert summary["updates"] == 4
    assert summary["mean_reward_first"] == pytest.approx(0.0)
    assert summary["mean_reward_last"] == pytest.approx(0.3)
    assert summary["mean_reward_max"] == pytest.approx(0.3)


def test_report_is_deterministic_and_written(tmp_path):
    records = [
        build_update_record(update_index=0, arm="e0", seed=42, rewards=[0.0, 1.0]).to_dict()
    ]
    report_a = build_report(experiment="x", arms={"e0": records})
    report_b = build_report(experiment="x", arms={"e0": records})
    assert report_a == report_b

    path = write_report(report_a, tmp_path)
    assert path.is_file()
    assert "e0" in path.read_text(encoding="utf-8")


def test_report_includes_pass_at_k(tmp_path):
    records = [build_update_record(update_index=0, arm="e0", seed=42, rewards=[1.0]).to_dict()]
    report = build_report(
        experiment="x",
        arms={"e0": records},
        per_task_rewards={"e0": {"airline_1": [1.0, 0.0]}},
    )
    assert "pass_at_k" in report
    assert report["pass_at_k"]["e0"]["pass@1"] == pytest.approx(0.5)
