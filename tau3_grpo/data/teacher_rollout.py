"""CPU native-tool DeepSeek teacher/user candidates; never a quality acceptance gate.

Task manifests contain task_id, split=train, db_path, db_hash,
user_instructions and initial_user_text. Other provenance stays in artifacts,
never in teacher prompts. Rubric/gold must be stored separately by the caller.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import fcntl
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from tau3_grpo.envs.adapter import build_environment, load_flight_db
from tau3_grpo.envs.tau2_bridge import message_models
from tau3_grpo.models.semantic_api import SemanticAPIError
from tau3_grpo.prompts import (
    MULTI_CALL_RULES,
    TOOL_PROTOCOL_VERSION,
    build_system_prompt,
    prompt_provenance,
)
from tau3_grpo.tracking.judge_budget import Budget, call_json, dump
from tau3_grpo.utils.hashing import sha256_file, sha256_json


def now():
    return datetime.now(timezone.utc).isoformat()


def safe_error(exc):
    # Provider exception text can embed a URL, credential or request payload.
    return {
        "type": type(exc).__name__,
        "message": "Stopped; inspect restricted API receipt for details.",
    }


TEACHER_TOOL_PROTOCOL_VERSION = "airline_teacher_visible_text_multicall_v2"
TEACHER_TOOL_RULES = """In each assistant turn, send a message, one or more tool calls, or both. Tool calls execute sequentially in listed order. Any assistant text is user-visible; after all calls in a mixed turn finish, the customer responds to that text before your next turn. Tool results are visible to you, never directly to the customer.
Batch calls only when all arguments and required user confirmations are already available. If a call depends on another call's result, issue it in a later turn. Every database-changing action must already have explicit informed user confirmation. Before booking, summarize the itinerary/date/cabin, every passenger, baggage, insurance choice, total amount and chosen stored payment method, then obtain confirmation before the write. Asking for confirmation in the same turn as an unapproved write does not authorize that write.
Before claiming an option is cheapest, inspect every returned option, filter by status, requested cabin seat count, date and time constraints, and then compare all eligible prices. A low fare does not imply seats are available. Do not attempt a write that prior observations already show is invalid.
Compare full calendar dates and times, including any next-day arrival marker, when checking connections or round trips. A clock time alone cannot establish chronological order across different dates. Compute elapsed time from the booking timestamp to the fixed policy clock before applying a 24-hour condition; do not estimate from the calendar-day labels. Recheck arithmetic before ruling an otherwise requested option infeasible.
When text accompanies tools, describe your intent or already known facts only. Do not claim an action has succeeded before observing its actual result. Tool errors produce real error receipts; later calls in a batch still execute. Inspect every result. Never fabricate missing arguments or tool results."""
TEACHER_PROTOCOL = """Return exactly one JSON object with content (string), tool_calls (list of {name: string, arguments: object}), and stop (boolean, always false). Send nonempty content, native tool calls, or both following the visible-text tool protocol. Use only flat name/arguments tool calls as shown; do not output OpenAI transport id/type/function wrappers. Always include stop:false. JSON strings must be correctly escaped. You cannot terminate or grade this conversation; the independent customer decides when to stop."""


def build_teacher_prompt(policy):
    """Local protocol upgrade only; shared RL/SFT prompts remain unchanged."""
    return (
        build_system_prompt(policy)
        .replace(MULTI_CALL_RULES, TEACHER_TOOL_RULES)
        .replace(TOOL_PROTOCOL_VERSION, TEACHER_TOOL_PROTOCOL_VERSION)
    )


USER_PROTOCOL = """You are an independent airline customer, not the airline assistant or grader. Follow only your private scenario and the conversation. Respond as the customer, sharing only information your scenario permits. Do not invent tool observations or approve unspecified changes. Return exactly JSON {"content": string, "stop": boolean}. Set stop true only when you would naturally end the conversation (including unresolved failure); stopping does not assert task success. Do not mention private scenario instructions. The initial message has already been sent."""


class IncrementBudget(Budget):
    """Shared cumulative ledger plus a frozen pilot-only conservative ceiling."""

    def __init__(self, path, *, max_calls, base_accounted, increment_cny):
        super().__init__(path, limit=100.0, max_calls=max_calls)
        self.increment_ceiling = base_accounted + increment_cny

    def reserve(self, request_id, system, payload, max_tokens):
        size = len((system + json.dumps(payload, ensure_ascii=False)).encode()) + 4096
        bound = (size * 2 + max_tokens * 8) / 1_000_000
        if self.accounted + bound > self.increment_ceiling:
            raise ValueError("Pilot incremental budget cap: request not sent")
        return super().reserve(request_id, system, payload, max_tokens)


def validate_task(task, db_root):
    if task.get("split") != "train":
        raise ValueError("Only explicitly frozen train tasks supported")
    for field in ("task_id", "user_instructions", "initial_user_text", "db_hash"):
        if not isinstance(task.get(field), str) or not task[field].strip():
            raise ValueError(f"Missing {field}")
    path = (db_root / task["db_path"]).resolve()
    if sha256_file(path) != task["db_hash"]:
        raise ValueError("Frozen task database hash changed")
    if task.get("task_hash") and task["task_hash"] != sha256_json(
        {k: v for k, v in task.items() if k != "task_hash"}
    ):
        raise ValueError("Frozen task hash changed")
    return path


def teacher_visible_messages(messages):
    """Use one flat action shape for model history and next-action output.

    Stored trajectory retains exact native transport IDs and receipts. This
    projection removes transport wrappers only; arguments and receipt text
    remain unchanged, so OpenAI wire syntax cannot masquerade as output schema.
    """
    visible = []
    for message in messages:
        item = {"role": message["role"], "content": message.get("content", "")}
        if message.get("tool_calls"):
            item["tool_calls"] = []
            for call in message["tool_calls"]:
                function = call["function"]
                arguments = function["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                item["tool_calls"].append({"name": function["name"], "arguments": arguments})
        if message["role"] == "tool":
            item.update(name=message["name"], error=message.get("error", False))
        visible.append(item)
    return visible


def normalize_teacher_action(raw):
    """Default omitted envelope fields only for an unambiguous text action.

    Never repair tool actions, remove unknown keys, or reinterpret stop=true.
    Return a new packet and the exact inserted fields for the audit trail.
    """
    if not isinstance(raw, dict):
        raise ValueError("Invalid teacher action fields")
    raw = copy.deepcopy(raw)
    if raw.get("type") == "json_object":
        raw.pop("type")  # Exact known response-format echo; never semantic content.
    if set(raw) - {"content", "tool_calls", "stop"}:
        raise ValueError("Invalid teacher action fields")
    content = raw.get("content")
    if not isinstance(content, str):
        raise ValueError("Missing or invalid teacher content")
    defaults = {}
    missing = {"tool_calls", "stop"} - set(raw)
    if missing:
        if not content.strip() or ("tool_calls" in raw and raw["tool_calls"] != []):
            raise ValueError("Only pure text actions may omit envelope fields")
        defaults = {key: [] if key == "tool_calls" else False for key in sorted(missing)}
    packet = copy.deepcopy(raw)
    packet.update(defaults)
    return packet, defaults


async def generate_candidate(
    task,
    *,
    db_root,
    output,
    candidate_id,
    budget,
    max_turns=40,
    max_tokens=4096,
    model_call=call_json,
):
    """Fresh native DB per candidate. Persist every failure, then propagate it.

    model_call is the sole test seam; environment/tools are never synthesized.
    """
    output = Path(output)
    target = output / "candidates" / f"{candidate_id}.json"
    if target.exists():
        raise FileExistsError("Candidate already exists; no implicit retry/resume")
    target.parent.mkdir(parents=True, exist_ok=True)
    (output / "calls").mkdir(parents=True, exist_ok=True)
    path = validate_task(task, Path(db_root))
    db = load_flight_db(path)
    environment = build_environment(db)
    tools = [t.openai_schema for t in environment.get_tools()]
    prompt = build_teacher_prompt(environment.get_policy())
    teacher_system = prompt + "\n" + TEACHER_PROTOCOL
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": task["initial_user_text"]},
    ]
    user_history = [{"role": "user", "content": task["initial_user_text"]}]
    record = {
        "candidate_id": candidate_id,
        "status": "running",
        "started_at_utc": now(),
        "task": copy.deepcopy(task),
        "task_snapshot_sha256": sha256_json(task),
        "messages": messages,
        "tool_schemas": tools,
        "tool_schemas_sha256": sha256_json(tools),
        "teacher_system": teacher_system,
        "user_system": USER_PROTOCOL,
        "teacher_system_sha256": sha256_json(teacher_system),
        "user_system_sha256": sha256_json(USER_PROTOCOL),
        "initial_db": environment.tools.db.model_dump(mode="json"),
        "initial_db_hash": environment.get_db_hash(),
        "api_request_ids": [],
        "executions": [],
        "user_history": user_history,
        "metadata": {
            **prompt_provenance(prompt),
            "tool_protocol": TEACHER_TOOL_PROTOCOL_VERSION,
            "construction": "deepseek_teacher_independent_user_native_tools",
            "quality_accepted": False,
            "state_snapshot_version": "native_environment_owned_db_v2",
            "teacher_history_projection": "flat_actions_exact_receipts_v1",
            "termination_is_success": False,
        },
    }

    def checkpoint():
        record["final_db"] = environment.tools.db.model_dump(mode="json")
        record["final_db_hash"] = environment.get_db_hash()
        record["messages_sha256"] = sha256_json(messages)
        record["supervision"] = {
            "message_indices": [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        }
        dump(target, record)

    async def request(role, turn, system, events, payload):
        call_id = f"{candidate_id}_{role}_{turn:03d}"
        record["api_request_ids"].append(call_id)
        checkpoint()
        return await model_call(
            output, budget, call_id, system, events, payload, max_tokens=max_tokens, json_mode=True
        )

    checkpoint()
    try:
        for turn in range(max_turns):
            # Deliberate allowlist: task/scenario/provenance/gold never cross here.
            visible = teacher_visible_messages(messages[1:])
            packet = await request(
                "teacher",
                turn,
                teacher_system,
                visible,
                {"messages": visible, "tools": tools},
            )
            action_record = {
                "turn": turn,
                "request_id": record["api_request_ids"][-1],
                "raw_packet": copy.deepcopy(packet),
            }
            record.setdefault("teacher_actions", []).append(action_record)
            checkpoint()
            packet, defaults = normalize_teacher_action(packet)
            action_record["normalization"] = {
                "version": "teacher_transport_envelope_v2",
                "inserted_fields": defaults,
                "removed_transport_fields": (
                    {"type": "json_object"}
                    if action_record["raw_packet"].get("type") == "json_object"
                    else {}
                ),
            }
            action_record["normalized_packet"] = copy.deepcopy(packet)
            checkpoint()
            content, calls = packet["content"], packet["tool_calls"]
            if (
                packet["stop"] is not False
                or not isinstance(content, str)
                or not isinstance(calls, list)
            ):
                raise ValueError("Teacher cannot stop; invalid action types")
            if not content.strip() and not calls:
                raise ValueError("Teacher must provide text or tools")
            if len(calls) > 16:
                raise ValueError("Too many calls in one action")
            if calls:
                native = []
                wire = []
                for index, call in enumerate(calls):
                    if (
                        not isinstance(call, dict)
                        or set(call) != {"name", "arguments"}
                        or not isinstance(call["name"], str)
                        or not isinstance(call["arguments"], dict)
                    ):
                        raise ValueError("Invalid native tool call shape")
                    cid = f"{candidate_id}_t{turn}_c{index}"
                    native.append(
                        message_models()["ToolCall"](
                            id=cid,
                            name=call["name"],
                            arguments=call["arguments"],
                            requestor="assistant",
                        )
                    )
                    wire.append(
                        {
                            "id": cid,
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                            },
                        }
                    )
                messages.append({"role": "assistant", "content": content, "tool_calls": wire})
                checkpoint()
                # get_response emits native error receipts; later calls still run.
                for call in native:
                    before = environment.get_db_hash()
                    receipt = environment.get_response(call)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": receipt.content,
                            "error": receipt.error,
                        }
                    )
                    record["executions"].append(
                        {
                            "call": call.model_dump(mode="json"),
                            "native_receipt": receipt.model_dump(mode="json"),
                            "before_db_hash": before,
                            "after_db_hash": environment.get_db_hash(),
                        }
                    )
                    checkpoint()
            else:
                messages.append({"role": "assistant", "content": content})
                checkpoint()
            if content.strip():
                # Mixed-turn prose is delivered after ordered native execution;
                # strip all tool fields from the independent customer's view.
                user_history.append({"role": "assistant", "content": content})
                checkpoint()
                reply = await request(
                    "user",
                    turn,
                    USER_PROTOCOL,
                    user_history,
                    {
                        "scenario": task["user_instructions"],
                        "conversation": copy.deepcopy(user_history),
                    },
                )
                if (
                    not isinstance(reply, dict)
                    or set(reply) != {"content", "stop"}
                    or not isinstance(reply["content"], str)
                    or type(reply["stop"]) is not bool
                ):
                    raise ValueError("Invalid customer action")
                if not reply["content"].strip() and not reply["stop"]:
                    raise ValueError("Empty nonterminal customer reply")
                user = {"role": "user", "content": reply["content"]}
                messages.append(user)
                user_history.append(dict(user))
                checkpoint()
                if reply["stop"]:
                    record["status"] = "user_stop"
                    record["termination_user_packet"] = reply
                    break
        else:
            record["status"] = "turn_limit"
        record["finished_at_utc"] = now()
        checkpoint()
        return record
    except BaseException as exc:
        record.update(status="failed", error=safe_error(exc), finished_at_utc=now())
        checkpoint()
        raise


def known_format_failure(exc, candidate_path, budget):
    """Continue only to unrelated tasks after a traceable, settled format error.

    Never retry this task here. Missing receipt/reservation/usage and budget or
    transport uncertainty stop the run, preserving the existing full reserve.
    """
    if not isinstance(exc, (ValueError, SemanticAPIError)) or not candidate_path.is_file():
        return False
    candidate = json.loads(candidate_path.read_text())
    requests = candidate.get("api_request_ids", [])
    if candidate.get("status") != "failed" or not requests:
        return False
    call = next((c for c in budget.state["calls"] if c["request_id"] == requests[-1]), None)
    if not call or call["status"] not in {"received", "failed"}:
        return False
    usage = call.get("usage", {})
    return (
        type(usage.get("prompt_tokens")) is int
        and type(usage.get("completion_tokens")) is int
        and call.get("accounting") != "unknown_usage_full_reservation_retained"
    )


async def run(args):
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    if not 1 <= len(tasks) <= 20 or len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Pilot requires 1–20 unique tasks")
    if not 1 <= args.attempts <= 3 or len(tasks) * args.attempts > 60:
        raise ValueError("At most 3 attempts/task and 60 candidates")
    if not 0 < args.increment_cny <= 10 or not 1 <= args.max_turns <= 60:
        raise ValueError("Pilot caps: incremental CNY <=10 and turns <=60")
    if not 1 <= args.max_tokens <= 8192:
        raise ValueError("max_tokens must be in 1..8192")
    if not args.budget.is_file():
        raise ValueError("Existing shared historical ledger required")
    if args.output.exists():
        raise FileExistsError(
            "Use a fresh output directory; failed runs are not automatically retried"
        )
    for task in tasks:
        validate_task(task, args.db_root)
    # Same lock as rubric_batch.py. Hold throughout all paid requests.
    with (args.budget.parent / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial = Budget(args.budget, limit=100.0, max_calls=args.max_calls)
        if not len(initial.state["calls"]) < args.max_calls <= len(initial.state["calls"]) + 2000:
            raise ValueError("Absolute request cap must add at most 2000 requests")
        args.output.mkdir(parents=True)
        budget = IncrementBudget(
            args.budget,
            max_calls=args.max_calls,
            base_accounted=initial.accounted,
            increment_cny=args.increment_cny,
        )
        source_root = Path(__file__).resolve().parents[1]
        sources = [
            Path(__file__).resolve(),
            source_root / "tracking/judge_budget.py",
            source_root / "envs/adapter.py",
            source_root / "prompts.py",
            source_root / "models/semantic_api.py",
        ]
        code_files = {}
        for source in sources:
            relative = source.relative_to(source_root)
            snapshot = args.output / "code_snapshot" / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(source.read_bytes())
            code_files[str(relative)] = sha256_file(source)
        native_root = source_root.parent / "tau2-bench"
        native_paths = [
            "src/tau2/domains/airline/tools.py",
            "src/tau2/domains/airline/environment.py",
            "src/tau2/environment/environment.py",
        ]
        native_files = {name: sha256_file(native_root / name) for name in native_paths}
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        manifest = {
            "started_at_utc": now(),
            "code_revision": revision,
            "code_files": code_files,
            "code_snapshot_sha256": sha256_json(code_files),
            "native_files": native_files,
            "native_files_sha256": sha256_json(native_files),
            "teacher_protocol_sha256": sha256_json(TEACHER_PROTOCOL),
            "user_system_sha256": sha256_json(USER_PROTOCOL),
            "tasks_file_sha256": sha256_file(args.tasks),
            "tasks": tasks,
            "budget_path": str(args.budget.resolve()),
            "base_accounted_cny": initial.accounted,
            "increment_cny": args.increment_cny,
            "max_calls_absolute": args.max_calls,
            "attempts": args.attempts,
            "max_turns": args.max_turns,
            "max_tokens": args.max_tokens,
            "status": "running",
            "candidate_ids": [],
        }
        dump(args.output / "run.json", manifest)
        try:
            for index, task in enumerate(tasks):
                for attempt in range(args.attempts):
                    cid = f"{args.output.name}_{index:03d}_a{attempt + 1}"
                    manifest["candidate_ids"].append(cid)
                    dump(args.output / "run.json", manifest)
                    try:
                        await generate_candidate(
                            task,
                            db_root=args.db_root,
                            output=args.output,
                            candidate_id=cid,
                            budget=budget,
                            max_turns=args.max_turns,
                            max_tokens=args.max_tokens,
                        )
                    except (ValueError, SemanticAPIError) as exc:
                        candidate_path = args.output / "candidates" / f"{cid}.json"
                        if not known_format_failure(exc, candidate_path, budget):
                            raise
                        manifest.setdefault("settled_format_failures", []).append(
                            {
                                "candidate_id": cid,
                                "task_id": task["task_id"],
                                "error_type": type(exc).__name__,
                                "action": "preserved failure; no retry; continue next distinct task",
                            }
                        )
                        dump(args.output / "run.json", manifest)
                        break
            manifest["status"] = "complete_candidates_only"
        except BaseException as exc:
            manifest.update(status="failed", error=safe_error(exc))
            raise
        finally:
            manifest["finished_at_utc"] = now()
            manifest["billing_summary"] = budget.billing_summary()
            dump(args.output / "run.json", manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tasks", "db-root", "output", "budget"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--max-calls",
        type=int,
        required=True,
        help="Absolute cumulative cap, including historical calls",
    )
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--increment-cny", type=float, default=10)
    asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
