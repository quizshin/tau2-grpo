"""Read-only, exact-token audit of archived rollout JSONL; never calls a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from tau3_grpo.models.generation_guard import GenerationGuard, detect_repetition
from tau3_grpo.models.token_budget import complete_tool_envelopes


def object_field(row, name):
    value = row.get(name)
    return json.loads(value) if isinstance(value, str) else value


def scan_turn(ids, tokenizer, config, stride=8):
    """Replay prefixes, not just the final suffix; refine first detection exactly."""
    for stop in range(config.min_tokens, len(ids) + stride, stride):
        stop = min(stop, len(ids))
        event = detect_repetition(ids[:stop], tokenizer.decode(ids[:stop], skip_special_tokens=True), config)
        if event:
            for exact in range(max(config.min_tokens, stop - stride + 1), stop + 1):
                event = detect_repetition(ids[:exact], tokenizer.decode(ids[:exact], skip_special_tokens=True), config)
                if event:
                    return event
    return None


def audit_row(row, tokenizer, config):
    facts = object_field(row, "trajectory_facts_json")
    if not facts or not facts.get("capabilities", {}).get("exact_response_ids"):
        raise ValueError("Historical row lacks exact token facts")
    tokens = facts["tokens"]
    ids, mask = tokens["response_ids"], tokens["response_mask"]
    if len(ids) != len(mask):
        raise ValueError("Historical token/mask mismatch")
    replay = object_field(row, "mt_gtpo_replay_json") or {}
    process = object_field(row, "process_reward_json") or {}
    advantages = replay.get("turn_advantages")
    rewards = process.get("turn_rewards")
    returns = replay.get("turn_returns")
    coverage = [0] * len(ids)
    rows = []
    for k, turn in enumerate(facts["turns"]):
        start, end = turn["token_span"]
        if not 0 <= start < end <= len(ids) or not all(mask[start:end]) or any(coverage[start:end]):
            raise ValueError("Historical assistant span invalid or partially discarded")
        coverage[start:end] = [1] * (end - start)
        part = ids[start:end]
        text = tokenizer.decode(part, skip_special_tokens=True)
        event = scan_turn(part, tokenizer, config)
        tool = "<tool_call" in text
        category = ("repetition" if event else "partial_tool" if not complete_tool_envelopes(text)
                    else "tool_call" if tool else "length_prose" if turn.get("finish_reason") == "length"
                    else "ordinary_message")
        advantage = advantages[k] if advantages is not None else None
        rows.append({"step": row["step"], "task_id": row.get("task_id"),
                     **facts["identity"], "turn_index": k, "token_span": [start, end],
                     "generated_tokens": len(part), "category": category, "repetition": event,
                     "finish_reason": turn.get("finish_reason"), "parsed": turn.get("parsed"),
                     "termination_reason": row.get("termination_reason"), "outcome": row.get("score"),
                     "turn_reward": rewards[k] if rewards is not None else None,
                     "turn_return": returns[k] if returns is not None else None,
                     "advantage": advantage,
                     "positive_advantage_tokens": len(part) if advantage is not None and advantage > 0 else 0,
                     "negative_advantage_tokens": len(part) if advantage is not None and advantage < 0 else 0,
                     "advantage_token_sum": advantage * len(part) if advantage is not None else None,
                     "token_sha256": hashlib.sha256(json.dumps(part).encode()).hexdigest(),
                     "text": text, "sampling": turn.get("generation_sampling_parameters"),
                     "logprobs_available": tokens.get("response_logprobs") is not None})
    if coverage != mask:
        raise ValueError("Historical assistant spans do not cover generation mask")
    return rows


def run_audit(root, output, tokenizer_path, first=10, last=27, repeats=32):
    from tau3_grpo.evaluation.token_runtime import load_tokenizer, tokenizer_identity

    output.mkdir(parents=True, exist_ok=False)
    tokenizer = load_tokenizer(tokenizer_path)
    config = GenerationGuard(mode="observe", repeats=repeats)
    sources, summary = [], []
    seen = set()
    with (output / "turns.jsonl").open("w") as sink:
        for path in sorted(root.glob("*/*/rollouts/*.jsonl")):
            step = int(path.stem)
            if not first <= step <= last:
                continue
            digest = hashlib.sha256()
            counts = Counter()
            group_counts = Counter()
            positive = Counter()
            for line in path.open("rb"):
                digest.update(line)
                row = json.loads(line)
                if row["step"] != step:
                    raise ValueError("File step and row step disagree")
                turns = audit_row(row, tokenizer, config)
                identity = turns[0]["trajectory_id"]
                key = (str(path.parent.parent), step, identity)
                if key in seen:
                    raise ValueError("Duplicate trajectory identity")
                seen.add(key)
                group_counts[turns[0]["sample_group_uid"]] += 1
                counts["trajectories"] += 1
                counts["successes"] += abs(float(row["score"]) - 1) <= 1e-6
                counts["context_terminations"] += row.get("termination_reason") == "context_window_exceeded"
                repeats_here = [t for t in turns if t["repetition"]]
                counts["repetition_trajectories"] += bool(repeats_here)
                counts["successful_repetition_trajectories"] += bool(repeats_here) and abs(float(row["score"]) - 1) <= 1e-6
                for turn in turns:
                    turn["run"] = str(path.parent.parent.relative_to(root))
                    sink.write(json.dumps(turn, ensure_ascii=False, allow_nan=False) + "\n")
                    counts["turns"] += 1
                    counts[turn["category"]] += 1
                    counts["raw_length_turns"] += turn["finish_reason"] == "length"
                    counts["generated_tokens"] += turn["generated_tokens"]
                    if turn["repetition"]:
                        counts["potential_tail_tokens"] += turn["generated_tokens"] - turn["repetition"]["detected_at_tokens"]
                    positive[turn["category"]] += turn["positive_advantage_tokens"]
            if counts["trajectories"] != 64 or len(group_counts) != 8 or set(group_counts.values()) != {8}:
                raise ValueError(f"Unexpected candidate/group coverage: {path}")
            sources.append({"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size})
            summary.append({"run": str(path.parent.parent.relative_to(root)), "step": step,
                            **counts, "positive_advantage_tokens_by_category": dict(positive)})
    manifest = {"scope": "CPU exact-token historical audit; no model calls", "first_step": first, "last_step": last,
                "settings": asdict(config), "sources": sources, "summary": summary,
                "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[2])):
                                  hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                                      Path(__file__).resolve(),
                                      Path(__file__).resolve().parents[1] / "models/generation_guard.py")},
                "tokenizer": {"path": str(tokenizer_path), "files_sha256": tokenizer_identity(tokenizer_path)},
                "ppo_loss_contribution_available": False,
                "limitation": "Archived rollout facts do not contain each optimizer minibatch current log-prob; advantage sums are not PPO losses or gradients."}
    if len(sources) != 2 * (last - first + 1):
        raise ValueError("Missing run/step files")
    (output / "summary.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--first", type=int, default=10)
    parser.add_argument("--last", type=int, default=27)
    parser.add_argument("--repeats", type=int, default=32)
    args = parser.parse_args()
    result = run_audit(args.root, args.output, args.tokenizer, args.first, args.last, args.repeats)
    print(json.dumps({"files": len(result["sources"]), "trajectories": sum(s["trajectories"] for s in result["summary"])}))


if __name__ == "__main__":
    main()
