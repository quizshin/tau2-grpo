"""Derive structured anchor features from recorded tau2 messages.

Milestone D6: the anchor needs more than the DB hash, because two states with an
identical DB can differ in what the agent has learned and confirmed. These
extractors are pure functions over the recorded message list so they are
reproducible from a stored trajectory.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from tau3_grpo.algorithms.anchors.encoder import ObservationType

#: Read-only Airline tools. Calling one means the agent has acquired information.
KNOWN_INFO_TOOLS: tuple[str, ...] = (
    "get_user_details",
    "get_reservation_details",
    "search_direct_flight",
    "search_onestop_flight",
    "list_all_airports",
    "get_flight_status",
)

#: Write tools. Calling one successfully means a commitment was made.
CONFIRMATION_TOOLS: tuple[str, ...] = (
    "book_reservation",
    "update_reservation_flights",
    "update_reservation_passengers",
    "update_reservation_baggages",
    "cancel_reservation",
    "send_certificate",
)

_CONFIRM_PATTERNS = (
    re.compile(r"\bconfirm(?:ed|ing|ation)?\b", re.IGNORECASE),
    re.compile(r"\byes\b", re.IGNORECASE),
    re.compile(r"\bgo ahead\b", re.IGNORECASE),
    re.compile(r"\bproceed\b", re.IGNORECASE),
)


def _tool_calls(message: Any) -> list[Any]:
    calls = getattr(message, "tool_calls", None)
    return list(calls) if calls else []


def _role(message: Any) -> str:
    return str(getattr(message, "role", ""))


def known_info_mask(messages: Iterable[Any]) -> tuple[bool, ...]:
    """One bit per read tool: has the agent successfully called it yet?

    Ordered by `KNOWN_INFO_TOOLS`, so the mask is positional and stable.
    """

    seen: set[str] = set()
    pending: dict[str, str] = {}
    for message in messages:
        for call in _tool_calls(message):
            if call.name in KNOWN_INFO_TOOLS:
                pending[str(call.id)] = call.name
        if _role(message) == "tool":
            name = pending.pop(str(getattr(message, "id", "")), None)
            if name is not None and not getattr(message, "error", False):
                seen.add(name)
    return tuple(tool in seen for tool in KNOWN_INFO_TOOLS)


def confirmation_flags(messages: Iterable[Any]) -> tuple[str, ...]:
    """Legacy v1 keyword flags, retained only for historical replay.

    Live v2 anchors use scoped dialogue evidence instead of user_confirmed.
    """

    flags: set[str] = set()
    pending: dict[str, str] = {}
    for message in messages:
        for call in _tool_calls(message):
            if call.name in CONFIRMATION_TOOLS:
                pending[str(call.id)] = call.name
        if _role(message) == "tool":
            name = pending.pop(str(getattr(message, "id", "")), None)
            if name is not None and not getattr(message, "error", False):
                flags.add(f"committed:{name}")
        if _role(message) == "user":
            content = getattr(message, "content", None) or ""
            if any(pattern.search(content) for pattern in _CONFIRM_PATTERNS):
                flags.add("user_confirmed")
    return tuple(sorted(flags))


def policy_precondition_flags(messages: Iterable[Any]) -> tuple[str, ...]:
    """Policy preconditions the agent has satisfied so far.

    Airline policy requires identifying the user before acting on a reservation,
    and reading a reservation before modifying it.
    """

    flags: set[str] = set()
    mask = dict(zip(KNOWN_INFO_TOOLS, known_info_mask(messages), strict=True))
    if mask.get("get_user_details"):
        flags.add("user_identified")
    if mask.get("get_reservation_details"):
        flags.add("reservation_loaded")
    if mask.get("search_direct_flight") or mask.get("search_onestop_flight"):
        flags.add("flights_searched")

    had_error = any(
        _role(message) == "tool" and getattr(message, "error", False) for message in messages
    )
    if had_error:
        flags.add("recovering_from_tool_error")
    return tuple(sorted(flags))


def last_observation_type(messages: Iterable[Any]) -> ObservationType:
    """Type of the final recorded message, from the policy's point of view."""

    last = None
    for message in messages:
        last = message
    if last is None:
        return ObservationType.NONE
    role = _role(last)
    if role == "user":
        return ObservationType.USER
    if role == "tool":
        return (
            ObservationType.TOOL_ERROR
            if getattr(last, "error", False)
            else ObservationType.TOOL_OK
        )
    if role == "assistant":
        return ObservationType.ASSISTANT
    return ObservationType.NONE


def tool_error_count(messages: Iterable[Any]) -> int:
    """Number of failed tool responses, used for failure categorisation."""

    return sum(
        1 for message in messages if _role(message) == "tool" and getattr(message, "error", False)
    )
