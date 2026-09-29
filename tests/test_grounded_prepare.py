"""Repair split selection must exclude prior evaluation/training reference users."""

from types import SimpleNamespace

from tau3_grpo.data import grounded_repair as repair


def test_prepare_excludes_formal_reference_users_without_private_assets(monkeypatch, tmp_path):
    users = ["formal_reservation", "formal_action", "old_dev", "fresh_a", "fresh_b", "fresh_c"]
    reservations = {f"R{i}": {"user_id": uid} for i, uid in enumerate(users)}
    db = {"users": {uid: {} for uid in users}, "reservations": reservations}
    train = SimpleNamespace(
        task_id="source", split="train", db_hash="db",
        task={"reservation_ids": list(reservations)},
    )
    formal = SimpleNamespace(task={
        "reservation_ids": ["R0"],
        "evaluation_criteria": {"actions": [{"arguments": {"user_id": "formal_action"}}]},
    })

    def manifest(path):
        if "selection" in path:
            return []
        if path.startswith("data/manifests/rl_curriculum50_20260912/"):
            return [formal]
        return [train]

    monkeypatch.setattr(repair, "read_manifest", manifest)
    monkeypatch.setattr(repair, "read_rows", lambda path: {"old_dev"} if "validation" in path else set())
    monkeypatch.setattr(repair, "user_ids", set)
    monkeypatch.setattr(repair, "ArealTaskRecord", SimpleNamespace(
        model_validate=lambda task: SimpleNamespace(resolve_db_path=lambda root: "fake-db"),
    ))
    monkeypatch.setattr(repair, "raw_database", lambda path: db)
    monkeypatch.setattr(repair, "task_feature", lambda task: task)
    monkeypatch.setattr(repair, "KINDS", ("read_only",))
    monkeypatch.setattr(repair, "suitable", lambda *args: True)

    plan, entries = repair.prepare(tmp_path / "plan")
    expected_protected = {"formal_reservation", "formal_action", "old_dev"}
    assert set(plan["protected_users"]) == expected_protected
    assert entries == {"source": train}
    assert len(plan["validation"]) == 1
    assert len(plan["train"]) == 2
    assert {row["user_id"] for row in plan["train"] + plan["validation"]} == {
        "fresh_a", "fresh_b", "fresh_c",
    }
    assert set(plan["heldout_users"]).isdisjoint(expected_protected)
    assert (tmp_path / "plan/plan.json").exists()
