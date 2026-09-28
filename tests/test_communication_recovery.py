"""Regression for observed judge evidence corruption and saved rollout recovery."""

import json
from types import SimpleNamespace

import pytest

from tau3_grpo.evaluation.communication_contract import CommunicationUnresolved, validate_packet
from tau3_grpo.evaluation.communication_facts import materialize_event_evidence, visible_facts
from tau3_grpo.evaluation.recovery import collect_complete


def test_ids_extract_verbatim_and_reject_ellipsized_quote():
    events = [
        dict(
            event_id="m0000",
            role="assistant",
            content="Booking Details:\nCabin: economy\nTotal: $393",
        )
    ]
    p = dict(
        checks=[
            dict(
                id="g",
                status="pass",
                reason_code="evidence",
                reason="Disclosed",
                evidence_event_ids=["m0000"],
            )
        ],
        simulator_scope=dict(status="pass", evidence_event_ids=[], reason="In scope"),
    )
    resolved = materialize_event_evidence(p, events)
    assert resolved["checks"][0]["evidence"][0]["quote"] == events[0]["content"]
    assert validate_packet(resolved, events, [dict(id="g")])["passed"]
    p["checks"][0]["evidence"] = [dict(event_id="m0000", quote="Booking Details: ... Total: $393")]
    with pytest.raises(CommunicationUnresolved):
        materialize_event_evidence(p, events)


@pytest.mark.parametrize("ids", [["invented"], ["m0000", "m0000"], [None]])
def test_fabricated_or_duplicate_ids_fail(ids):
    p = dict(checks=[dict(evidence_event_ids=ids)], simulator_scope=dict(evidence_event_ids=[]))
    with pytest.raises(CommunicationUnresolved):
        materialize_event_evidence(p, [dict(event_id="m0000", role="assistant", content="ok")])


def test_tool_only_citation_still_does_not_prove_communication():
    e = [dict(event_id="m0000", role="tool", content='{"refund":2462}')]
    p = dict(
        checks=[
            dict(
                id="g",
                status="pass",
                reason_code="evidence",
                reason="Refund",
                evidence_event_ids=["m0000"],
            )
        ],
        simulator_scope=dict(status="pass", reason="In scope", evidence_event_ids=[]),
    )
    with pytest.raises(CommunicationUnresolved, match="assistant language"):
        validate_packet(materialize_event_evidence(p, e), e, [dict(id="g")])


@pytest.mark.parametrize(
    "old,new,n,expected",
    [
        ([154, 127], [463, 278], 2, 920),
        ([1750, 537], [133, 127], 3, -6081),
        ([966, 1096], [62, 87], 2, -3826),
    ],
)
def test_history_prices_times_passengers_and_incremental_payments(old, new, n, expected):
    def rec(prices, payments):
        return dict(
            reservation_id="R",
            flights=[dict(price=p) for p in prices],
            passengers=[{}] * n,
            payment_history=payments,
        )

    e = [
        dict(
            event_id="m0000",
            role="tool",
            error=False,
            content=json.dumps(rec(old, [dict(amount=100)])),
        ),
        dict(
            event_id="m0001",
            role="assistant",
            tool_calls=[dict(id="w", name="update_reservation_flights")],
        ),
        dict(
            event_id="m0002",
            role="tool",
            tool_call_id="w",
            error=False,
            content=json.dumps(rec(new, [dict(amount=100), dict(amount=expected)])),
        ),
    ]
    f = visible_facts(e)["facts"][-1]
    assert f["signed_fare_delta"] == f["signed_transaction_delta"] == expected
    assert f["booked_fare_total"] == sum(new) * n


@pytest.mark.parametrize(
    "created,inside",
    [("2024-05-14T14:45:00", False), ("2024-05-14T15:00:00", True), ("2024-05-15T15:00:01", False)],
)
def test_24_hour_boundary(created, inside):
    e = [
        dict(
            event_id="m0000",
            role="tool",
            error=False,
            content=json.dumps(
                dict(reservation_id="R", flights=[], passengers=[], created_at=created)
            ),
        )
    ]
    assert visible_facts(e)["facts"][0]["within_last_24_hours"] is inside


def test_complete_errors_are_reused_and_duplicates_rejected():
    job = dict(task_id="a", trial=0, seed=42)
    sim = dict(messages=[dict(role="assistant", content="done")])
    a = SimpleNamespace(
        metadata=dict(planned=[job]),
        trajectories=[],
        errors=[dict(job, execution_evidence=dict(simulation=sim))],
    )
    assert collect_complete(a)[0][1] == sim
    a.trajectories = [dict(job, simulation=sim)]
    with pytest.raises(ValueError, match="duplicate"):
        collect_complete(a)


def test_empty_missing_statement_remains_model_failure():
    p = dict(
        checks=[
            dict(
                id="g",
                status="fail",
                reason_code="missing",
                reason="Net charge absent",
                evidence_event_ids=[],
            )
        ],
        simulator_scope=dict(status="pass", reason="In scope", evidence_event_ids=[]),
    )
    assert not validate_packet(materialize_event_evidence(p, []), [], [dict(id="g")])["passed"]


def test_consent_after_write_remains_invalid():
    e = [
        dict(event_id="m0000", role="assistant", content="Charge $920?"),
        dict(event_id="m0001", role="assistant", content="Writing now."),
        dict(event_id="m0002", role="user", content="Yes"),
    ]
    p = dict(
        checks=[
            dict(
                id="consent_w",
                status="pass",
                reason_code="evidence",
                reason="Confirmed",
                evidence_event_ids=["m0000", "m0002"],
            )
        ],
        simulator_scope=dict(status="pass", reason="In scope", evidence_event_ids=[]),
    )
    with pytest.raises(CommunicationUnresolved, match="precede"):
        validate_packet(
            materialize_event_evidence(p, e), e, [dict(id="consent_w", call=dict(event_index=1))]
        )


def test_budget_cannot_reset_on_recovery():
    from tau3_grpo.evaluation.bounded import validate_plan

    plan = dict(
        version="bounded_dual_selection_v1",
        trials=1,
        seed=42,
        task_count=60,
        wall_seconds=7200,
        prior_wall_seconds=714,
        gpus=[0, 1],
        arms=[dict(name=x) for x in ("base", "sft1", "sft3")],
        training=False,
        formal_evaluation=False,
        source_sha256={"x": "hash"},
        authorization_sha256="hash",
    )
    with pytest.raises(ValueError, match="original combined"):
        validate_plan(plan)
    plan["wall_seconds"] = 6480
    validate_plan(plan)


def test_explicit_status_alias_preserves_raw_and_does_not_relax_evidence():
    e = [dict(event_id="m0000", role="assistant", content="I cannot determine the reason.")]
    raw = dict(
        checks=[
            dict(
                id="g",
                status_code="pass",
                reason_code="evidence",
                reason="Explained",
                evidence_event_ids=["m0000"],
            )
        ],
        simulator_scope=dict(status="pass", reason="In scope", evidence_event_ids=[]),
    )
    result = validate_packet(materialize_event_evidence(raw, e), e, [dict(id="g")])
    assert result["passed"] and result["status_alias_normalizations"] == ["g"]
    assert "status" not in raw["checks"][0]
    raw["checks"][0]["status"] = "fail"
    with pytest.raises(CommunicationUnresolved, match="Ambiguous status"):
        validate_packet(materialize_event_evidence(raw, e), e, [dict(id="g")])
    raw["checks"][0].pop("status")
    raw["checks"][0]["evidence_event_ids"] = []
    with pytest.raises(CommunicationUnresolved, match="positive evidence"):
        validate_packet(materialize_event_evidence(raw, e), e, [dict(id="g")])
