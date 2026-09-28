"""Local native integration guards for the composed data-only probe."""

import pytest

from tau3_grpo.data.grounded_repair import (
    Builder,
    build_cabin_then_baggage,
    build_curriculum_strategy,
)
from tau3_grpo.data.manifest import read_manifest


def entry():
    return next(
        e
        for e in read_manifest("data/manifests/areal_airline_train_seed42.jsonl")
        if e.task_id == "airline_497"
    )


def test_composed_probe_uses_new_allowance_and_confirmations():
    row = build_cabin_then_baggage(entry(), "DX8C9F")
    messages = row["messages"]
    writes = [
        (i, m["tool_calls"][0]["function"])
        for i, m in enumerate(messages)
        if m.get("tool_calls") and m["tool_calls"][0]["function"]["name"].startswith("update_")
    ]
    assert [c["name"] for _, c in writes] == [
        "update_reservation_flights",
        "update_reservation_baggages",
    ]
    assert writes[1][1]["arguments"]["nonfree_baggages"] == 1  # old cabin would charge 2
    assert all(
        messages[i - 1]["role"] == "user" and messages[i - 1]["content"].startswith("Yes")
        for i, _ in writes
    )
    assert row["metadata"]["baggage_checks"][0]["status"] == "satisfied"


def test_full_database_guard_rejects_unrequested_state(monkeypatch):
    original = Builder.finish

    def changed_expectation(self, reason, checks):
        self.expected["reservations"][self.rid]["insurance"] = "yes"
        return original(self, reason, checks)

    monkeypatch.setattr(Builder, "finish", changed_expectation)
    with pytest.raises(ValueError, match="protected-state predicate failed"):
        build_cabin_then_baggage(entry(), "DX8C9F")


@pytest.mark.parametrize(
    "kind,task_id,rid",
    [
        ("upgrade_then_bags", "airline_497", "DX8C9F"),
        ("gift_limit_upgrade_bags", "airline_615", "AEEF40"),
        ("downgrade_gift_bags", "airline_808", "ZA19EA"),
        ("partial_change_bags", "airline_1079", "0D8914"),
        ("no_inventory_cabin_bags", "airline_620", "IS6JKD"),
        ("cancel_refund_rebook", "airline_808", "ZA19EA"),
    ],
)
def test_curriculum_strategy_native_contracts(kind, task_id, rid):
    source = next(
        e
        for e in read_manifest("data/manifests/areal_airline_train_seed42.jsonl")
        if e.task_id == task_id
    )
    row = build_curriculum_strategy(source, rid, kind)
    messages = row["messages"]
    writes = []
    for i, message in enumerate(messages):
        for call in message.get("tool_calls", []):
            name = call["function"]["name"]
            if name.startswith("update_") or name in ("cancel_reservation", "book_reservation"):
                writes.append(name)
                assert messages[i - 1]["role"] == "user"
                assert messages[i - 1]["content"].startswith("Yes")
    assert len(writes) == 2
    assert row["metadata"]["outcome_sha256"]
    assert all(c["status"] == "satisfied" for c in row["metadata"]["baggage_checks"])


@pytest.mark.parametrize(
    "kind",
    [
        "book_direct_credit",
        "book_direct_split",
        "book_roundtrip_split",
        "book_date_replan",
        "book_connection_split",
    ],
)
def test_search_booking_receipts_payment_and_confirmation(kind):
    from tau3_grpo.data.grounded_repair import (
        build_curriculum_search,
        curriculum_pool,
        curriculum_search_option,
    )

    pool, entries, dbs, _, _ = curriculum_pool()
    item = next(x for x in pool if curriculum_search_option(dbs[x["task_id"]], x["rid"], kind))
    row = build_curriculum_search(entries[item["task_id"]], item["rid"], kind)
    messages = row["messages"]
    writes = []
    for i, message in enumerate(messages):
        for call in message.get("tool_calls", []):
            if call["function"]["name"] == "book_reservation":
                args = call["function"]["arguments"]
                writes.append(args)
                assert messages[i - 1]["role"] == "user" and messages[i - 1]["content"].startswith(
                    "Yes"
                )
                assert all(p["amount"] > 0 for p in args["payment_methods"])
                assert (
                    sum(p["amount"] for p in args["payment_methods"])
                    == curriculum_search_option(dbs[item["task_id"]], item["rid"], kind)["total"]
                )
                assert all(
                    f["flight_number"] not in messages[1]["content"] for f in args["flights"]
                )
    assert len(writes) == 1
    if "split" in kind:
        assert len(writes[0]["payment_methods"]) == 2
    if kind == "book_roundtrip_split":
        assert len(writes[0]["flights"]) == 2
    if kind == "book_connection_split":
        pairs = [
            __import__("json").loads(m["content"])
            for m in messages
            if m["role"] == "tool" and m.get("name") == "search_onestop_flight"
        ]
        assert len(pairs) == 1 and len(writes[0]["flights"]) == 2
        assert "one-stop" in messages[1]["content"]
    else:
        assert "nonstop" in messages[1]["content"]
    if kind == "book_date_replan":
        assert any("do not meet your seat" in m.get("content", "") for m in messages)


def test_historical_receipt_only_user_is_not_unexposed():
    from tau3_grpo.data.grounded_repair import user_ids

    rows = [
        {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_reservation_details",
                                "arguments": {"reservation_id": "R1"},
                            }
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"reservation_id":"R1","user_id":"receipt_only_user"}',
                },
            ]
        }
    ]
    assert user_ids(rows) == {"receipt_only_user"}


def test_synthetic_database_date_keys_are_not_valid_calendar_goals():
    from tau3_grpo.data.grounded_repair import audit_curriculum_dates, curriculum_valid_date

    assert curriculum_valid_date("2024-05-25")
    assert not curriculum_valid_date("2024-05-25_late")
    assert not curriculum_valid_date("2024-02-30")
    messages = [
        {"role": "user", "content": "Travel on 2024-05-25_late."},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "search_direct_flight", "arguments": {"date": "2024-02-30"}}}
            ],
        },
    ]
    assert audit_curriculum_dates(messages) == ["2024-02-30", "2024-05-25_late"]


def test_finalized_curriculum_cannot_be_appended(tmp_path):
    from types import SimpleNamespace

    from tau3_grpo.data.grounded_repair import run_curriculum

    (tmp_path / "summary.json").write_text("{}")
    with pytest.raises(FileExistsError, match="immutable"):
        run_curriculum(SimpleNamespace(output=tmp_path))
