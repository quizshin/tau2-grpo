"""Group-preserving bucket splits for reviewed labels; no trajectory-based difficulty.

This produces a candidate manifest only. Historical exposures remain locked, and
unknown labels or group identity block a row instead of silently assigning it easy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

BUCKETS = ("foundation", "constraints", "strategy")


def assign_bucket(label):
    if label.get("label_status") != "reviewed" or label.get("difficulty") not in (
        "easy",
        "medium",
        "hard",
    ):
        return None
    if label.get("difficulty_basis") != "task_constraints":
        return None
    for key in ("needs_replanning", "needs_dependent_subgoals"):
        if type(label.get(key)) is not bool:
            return None
    if (
        label["difficulty"] == "hard"
        or label["needs_replanning"]
        or label["needs_dependent_subgoals"]
    ):
        return "strategy"
    return "foundation" if label["difficulty"] == "easy" else "constraints"


def build_split(labels, *, seed=42, validation_ratio=0.1, protected_groups=()):
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between zero and one")
    ids = [x["source_id"] for x in labels]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source IDs")
    protected = set(protected_groups)
    groups, held = defaultdict(list), []
    for label in labels:
        group = label.get("group_id")
        if not group or label.get("group_status") != "reviewed":
            held.append(dict(source_id=label["source_id"], reason="unreviewed_group"))
            continue
        groups[group].append(label)
    splits = {"train": [], "validation": []}
    available = defaultdict(list)
    group_assignments, notes, targets = {}, [], {}
    for group, members in sorted(groups.items()):
        locks = {m.get("locked_split") for m in members} - {None}
        if locks - {"train", "validation"}:
            raise ValueError(f"unknown historical split in {group}")
        if len(locks) > 1:
            raise ValueError(f"historical train/dev family collision: {group}")
        if group in protected:
            held.extend(dict(source_id=m["source_id"], reason="protected_group") for m in members)
            continue
        buckets = [assign_bucket(m) for m in members]
        if any(b is None for b in buckets):
            held.extend(
                dict(source_id=m["source_id"], reason="unreviewed_difficulty_in_group")
                for m in members
            )
            continue
        bucket = max(buckets, key=BUCKETS.index)
        if len(set(buckets)) > 1:
            notes.append(
                dict(group_id=group, reason="mixed_bucket_group_kept_together", bucket=bucket)
            )
        available[bucket].append((group, members, next(iter(locks), None)))
    for bucket in BUCKETS:
        entries = available[bucket]
        total = sum(len(m) for _, m, _ in entries)
        target = int(total * validation_ratio + 0.5)
        current = sum(len(m) for _, m, lock in entries if lock == "validation")
        targets[bucket] = {"rows": total, "validation_target": target}
        for group, members, lock in entries:
            if lock:
                splits[lock].extend(m["source_id"] for m in members)
                group_assignments[group] = lock
        free = sorted(
            (e for e in entries if e[2] is None),
            key=lambda e: hashlib.sha256(f"{seed}:{e[0]}".encode()).hexdigest(),
        )
        for group, members, _ in free:
            # Approximate 10% by whole groups, never split a family to fill a quota.
            choose_dev = abs(current + len(members) - target) < abs(current - target)
            dest = "validation" if choose_dev else "train"
            current += len(members) if choose_dev else 0
            splits[dest].extend(m["source_id"] for m in members)
            group_assignments[group] = dest
        targets[bucket]["validation_actual"] = current
        targets[bucket]["train_actual"] = total - current
        if current != target:
            notes.append(
                dict(bucket=bucket, reason="group_size_or_historical_lock_prevents_exact_ratio")
            )
    for values in splits.values():
        values.sort()
    return dict(
        version="airline_reviewed_group_bucket_v1",
        status="candidate_manifest_not_training_authorization",
        seed=seed,
        validation_ratio=validation_ratio,
        splits=splits,
        group_assignments=group_assignments,
        buckets=targets,
        held=sorted(held, key=lambda h: h["source_id"]),
        notes=notes,
        held_reasons=dict(Counter(h["reason"] for h in held)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--protected-groups",
        type=Path,
        required=True,
        help="JSON array of precomputed holdout group IDs, not task answers",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    args = parser.parse_args()
    labels = [
        json.loads(lint_item)
        for lint_item in args.labels.read_text().splitlines()
        if lint_item.strip()
    ]
    result = build_split(
        labels,
        seed=args.seed,
        validation_ratio=args.validation_ratio,
        protected_groups=json.loads(args.protected_groups.read_text()),
    )
    result["inputs_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (args.labels, args.protected_groups)
    }
    with args.output.open("x") as output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
