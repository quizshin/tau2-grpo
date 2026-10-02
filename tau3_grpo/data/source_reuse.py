"""Materialize source-first reuse queues with existing evidence, no paid calls.

Does not silently turn old judge labels into intrinsic difficulty or training
approval. Preserves complete conversations and original assistant target mask.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from tau3_grpo.data.messages import visible_events
from tau3_grpo.data.sft import load_complete_airline_dialogues
from tau3_grpo.utils.hashing import sha256_file, sha256_json


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def evidence_ledger(config, reviews):
    """Merge evidence without promoting membership, execution, or labels to approval.

    Exact deduplication includes the supervision mask. All source aliases, content
    variants and historical memberships survive. Template overlap is a diagnostic,
    not proof of leakage or a reason to discard accepted training examples.
    """
    from tau3_grpo.data.curriculum_split import BUCKETS, assign_bucket, build_split
    from tau3_grpo.data.finalize_staged_sft import visible_user_ids
    from tau3_grpo.data.sft_evidence import digest as legacy_audit_digest
    from tau3_grpo.data.staged_sft import ordered_tool_receipts

    records, payloads, hashes = {}, {}, {}
    source_variants = defaultdict(set)
    for dataset in config["datasets"]:
        path = Path(dataset["path"])
        digest = sha256_file(path)
        if digest != dataset["sha256"]:
            raise ValueError(f"Input changed: {path}")
        hashes[str(path)] = digest
        for line, row in enumerate(read_jsonl(path), 1):
            sid = row["metadata"]["source_dialog_id"]
            messages_hash = sha256_json(row["messages"])
            indices = row.get("supervision", {}).get("message_indices")
            if not isinstance(indices, list) or not indices or len(set(indices)) != len(indices):
                raise ValueError(f"Invalid supervision mask: {sid}")
            if any(
                type(i) is not int
                or not 0 <= i < len(row["messages"])
                or row["messages"][i]["role"] != "assistant"
                for i in indices
            ):
                raise ValueError(f"Non-assistant supervision: {sid}")
            key = sha256_json(dict(messages=row["messages"], message_indices=indices))
            source_variants[sid].add(messages_hash)
            if key not in records:
                records[key] = dict(
                    source_id=key,
                    messages_sha256=messages_hash,
                    supervision_sha256=sha256_json(indices),
                    source_aliases=[],
                    memberships=[],
                    entities=[],
                    source_tasks=[],
                    source_dbs=[],
                    known_template_families=[],
                    evidence=[],
                    complete_dialogue=True,
                    new_semantic_review=None,
                )
            item = records[key]
            metadata = row["metadata"]
            item["source_aliases"].append(sid)
            item["memberships"].append(
                dict(
                    dataset=dataset["name"],
                    path=str(path),
                    line=line,
                    source_id=sid,
                    locked_split=dataset.get("locked_split"),
                    exposure=dataset["exposure"],
                    exposure_basis=dataset["exposure_basis"],
                )
            )
            item["entities"].extend(visible_user_ids(row))
            if metadata.get("source_task_id"):
                item["source_tasks"].append(metadata["source_task_id"])
            if metadata.get("source_db_hash"):
                item["source_dbs"].append(metadata["source_db_hash"])
            kind = metadata.get("repair_kind") or metadata.get("construction_kind")
            if kind:
                item["known_template_families"].append("authored:" + kind)
            item["evidence"].append(
                dict(
                    dataset=dataset["name"],
                    source_id=sid,
                    status=metadata.get("status"),
                    evidence_level=metadata.get("evidence_level"),
                    verification=metadata.get("verification"),
                    prior_review_sha256=metadata.get("current_review_sha256")
                    or metadata.get("review_sha256"),
                    adjudication=metadata.get("adjudication"),
                    stored_execution_count=len(metadata.get("executions", [])),
                    ordered_tool_receipt_errors=ordered_tool_receipts(row),
                    historical_label_tokens=metadata.get("label_tokens"),
                    inherited_difficulty_vector=metadata.get("difficulty_vector_uncalibrated"),
                    inherited_trajectory_difficulty=metadata.get("difficulty"),
                    semantic_checks=metadata.get("semantic_checks", []),
                    exact_task_db_mapping=bool(metadata.get("source_db_hash")),
                )
            )
            # Later configured versions take precedence for metadata only; identity
            # and assistant positions are identical by construction.
            payloads[key] = row

    decisions_path = Path(config["decisions"])
    hashes[str(decisions_path)] = sha256_file(decisions_path)
    decisions = {d["id"]: d for d in read_jsonl(decisions_path)}
    token_index = {}
    for spec in config.get("token_sources", []):
        path = Path(spec["path"])
        hashes[str(path)] = sha256_file(path)
        token_rows = json.loads(path.read_text())
        if isinstance(token_rows, dict) and token_rows.get("source_sha256"):
            dataset = next(d for d in config["datasets"] if d["name"] == spec["dataset"])
            if token_rows["source_sha256"] != dataset["sha256"]:
                raise ValueError("Token count source identity mismatch")
        for part in spec["selector"]:
            token_rows = token_rows[part]
        for entry in token_rows:
            token_index[(spec["dataset"], entry[spec["id_field"]])] = dict(
                entry, count_basis=spec.get("count_basis", "historical_identical_messages_and_mask")
            )
    legacy_labels = defaultdict(list)
    if config.get("legacy_labels"):
        path = Path(config["legacy_labels"])
        hashes[str(path)] = sha256_file(path)
        for label in read_jsonl(path):
            legacy_labels[(label["source_id"], label["messages_sha256"])].append(label)
    review_index = {(r["source_id"], r["messages_sha256"]): r for r in reviews}
    if len(review_index) != len(reviews):
        raise ValueError("Duplicate semantic review identity")
    used_reviews = set()
    groups = defaultdict(list)
    for key, item in records.items():
        for field in (
            "source_aliases",
            "entities",
            "source_tasks",
            "source_dbs",
            "known_template_families",
        ):
            item[field] = sorted(set(item[field]))
        item["prior_adjudications"] = [
            decisions[s] for s in item["source_aliases"] if s in decisions
        ]
        # The old audit hashes ensure_ascii=True JSON; canonical_json here uses
        # ensure_ascii=False. Compare its own serialization, never source ID alone.
        legacy_messages_hash = legacy_audit_digest(payloads[key]["messages"])
        item["inherited_curriculum_labels"] = [
            lint_item
            for s in item["source_aliases"]
            for lint_item in legacy_labels[(s, legacy_messages_hash)]
        ]
        item["inherited_label_hash_basis"] = "sft_coldstart_audit.digest_ensure_ascii_true"
        item["render_token_counts"] = [
            dict(dataset=m["dataset"], **token_index[(m["dataset"], m["source_id"])])
            for m in item["memberships"]
            if (m["dataset"], m["source_id"]) in token_index
        ]
        item["historical_render_token_counts"] = [
            t
            for t in item["render_token_counts"]
            if t["count_basis"] == "historical_identical_messages_and_mask"
        ]
        for decision in item["prior_adjudications"]:
            review_path = Path(config["source_review_records"]) / (decision["id"] + ".json")
            prior_review = json.loads(review_path.read_text())
            visible_hash = sha256_json(visible_events(payloads[key]["messages"]))
            if (
                visible_hash != decision["review_evidence_sha256"]
                or visible_hash != prior_review["evidence_sha256"]
                or sha256_file(review_path) != decision["review_record_sha256"]
            ):
                raise ValueError("Prior adjudication evidence mismatch: " + decision["id"])
            hashes[str(review_path)] = sha256_file(review_path)
        matched = [
            review_index[(s, item["messages_sha256"])]
            for s in item["source_aliases"]
            if (s, item["messages_sha256"]) in review_index
        ]
        if len(matched) > 1:
            raise ValueError("Multiple reviews for identical content aliases")
        if matched:
            for evidence in matched[0]["evidence"]:
                if not evidence["message_indices"] or any(
                    type(i) is not int or not 0 <= i < len(payloads[key]["messages"])
                    for i in evidence["message_indices"]
                ):
                    raise ValueError("Invalid semantic evidence indices")
            item["new_semantic_review"] = matched[0]
            used_reviews.add((matched[0]["source_id"], item["messages_sha256"]))
        locks = sorted({m["locked_split"] for m in item["memberships"] if m["locked_split"]})
        item["historical_split_locks"] = locks
        item["historical_gradient_exposure"] = any(
            m["exposure"] == "gradient_confirmed" for m in item["memberships"]
        )
        item["historical_dev_exposure"] = any(
            m["exposure"] == "diagnostic_dev" for m in item["memberships"]
        )
        for family in item["known_template_families"]:
            groups[family].append(key)
        item["semantic_quality"] = (
            matched[0]["verdict"]
            if matched
            else "inherited_experimental_acceptance"
            if any(d["decision"] == "include_experimental" for d in item["prior_adjudications"])
            else "historical_authored_execution_candidate"
            if item["known_template_families"]
            else "unreviewed_or_prior_hold"
        )
    if used_reviews != set(review_index):
        raise ValueError("Review does not match current source content")

    overlaps = []
    for family, keys in sorted(groups.items()):
        locks = sorted({lock for key in keys for lock in records[key]["historical_split_locks"]})
        if len(locks) > 1:
            overlaps.append(
                dict(
                    template=family,
                    source_ids=sorted(keys),
                    historical_splits=locks,
                    interpretation="template_overlap_not_semantic_leakage_verdict",
                )
            )
    entity_groups = defaultdict(list)
    for key, item in records.items():
        for entity in item["entities"]:
            entity_groups[entity].append(key)
    entity_collisions = [
        dict(entity=entity, source_ids=sorted(keys))
        for entity, keys in sorted(entity_groups.items())
        if len({lock for key in keys for lock in records[key]["historical_split_locks"]}) > 1
    ]
    labels = []
    for key, item in sorted(records.items()):
        review = item["new_semantic_review"] or {}
        # All corpus-wide family reviews remain pending unless explicit evidence
        # establishes that review. A specific task review is not a corpus review.
        label = dict(
            source_id=key,
            messages_sha256=item["messages_sha256"],
            source_aliases=item["source_aliases"],
            locked_split=(
                item["historical_split_locks"][0]
                if len(item["historical_split_locks"]) == 1
                else None
            ),
            group_id=review.get("family"),
            group_status="pending_corpus_family_review",
            label_status="reviewed" if review.get("difficulty") else "pending",
            difficulty=review.get("difficulty"),
            difficulty_basis="task_constraints",
            needs_replanning=review.get("needs_replanning"),
            needs_dependent_subgoals=review.get("needs_dependent_subgoals"),
            quality_status=item["semantic_quality"],
        )
        item["independent_dev_blockers"] = ["corpus_semantic_family_review_pending"]
        item["training_candidate_blockers"] = []
        if len(item["historical_split_locks"]) > 1:
            item["training_candidate_blockers"].append("exact_content_train_dev_collision")
        if item["historical_split_locks"] != ["train"]:
            item["training_candidate_blockers"].append("not_locked_training_membership")
        accepted_membership = any(
            m["dataset"] in config.get("accepted_training_cohorts", []) for m in item["memberships"]
        )
        newly_accepted = review.get("verdict") == "pass_visible_evidence" and any(
            m["dataset"] in config.get("new_review_required_cohorts", [])
            for m in item["memberships"]
        )
        item["acceptance_basis"] = (
            "inherited_cohort_acceptance"
            if accepted_membership
            else "new_full_dialogue_semantic_review"
            if newly_accepted
            else "none"
        )
        accepted_membership = accepted_membership or newly_accepted
        if not accepted_membership:
            item["training_candidate_blockers"].append("no_inherited_full_dialogue_acceptance")
        if review.get("verdict", "").startswith("hold"):
            item["training_candidate_blockers"].append("new_semantic_hold")
        if any(e["ordered_tool_receipt_errors"] for e in item["evidence"]):
            item["training_candidate_blockers"].append("tool_receipt_structure_error")
        item["training_ready"] = not item["training_candidate_blockers"]
        item["training_eligibility_scope"] = "experimental_data_candidate_not_new_run_authorization"
        item["independent_dev_ready"] = False
        labels.append(label)
    manifest = build_split(labels, seed=42, protected_groups=())
    training_labels = [
        dict(
            lint_item,
            group_status="reviewed",
            group_review_scope="current_semantically_reviewed_training_subset_only",
        )
        for lint_item in labels
        if records[lint_item["source_id"]]["training_ready"]
        and records[lint_item["source_id"]]["semantic_quality"] == "pass_visible_evidence"
    ]
    training_manifest = build_split(training_labels, seed=42, protected_groups=())
    training_manifest["scope"] = "historically_locked_training_subset_not_independent_validation"
    training_manifest["independent_family_dev_ready"] = False
    previews = {}
    for index, bucket in enumerate(BUCKETS):
        # Training membership is already locked. An unavailable independent dev
        # must not prevent materializing accepted, difficulty-reviewed train rows.
        previews[bucket] = [
            payloads[lint_item["source_id"]]
            for lint_item in labels
            if records[lint_item["source_id"]]["semantic_quality"] == "pass_visible_evidence"
            and records[lint_item["source_id"]]["training_ready"]
            and assign_bucket(lint_item) in BUCKETS[: index + 1]
            and lint_item["locked_split"] == "train"
        ]
    return dict(
        records=list(records.values()),
        labels=labels,
        manifest=manifest,
        previews=previews,
        payloads=payloads,
        template_overlap=overlaps,
        reviewed_training_manifest=training_manifest,
        entity_collisions=entity_collisions,
        source_content_variants=[
            dict(source_id=s, messages_sha256=sorted(v))
            for s, v in sorted(source_variants.items())
            if len(v) > 1
        ],
        inputs_sha256=hashes,
    )


def run_ledger(args):
    """Versioned evidence materialization; never writes a training authorization."""
    if args.output.exists():
        raise FileExistsError("Never overwrite an evidence inventory")
    config = json.loads(args.ledger_config.read_text())
    result = evidence_ledger(config, read_jsonl(args.semantic_reviews))
    args.output.mkdir(parents=True)

    def save(name, rows):
        (args.output / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )

    save("ledger.jsonl", result["records"])
    save("curriculum_labels.jsonl", result["labels"])
    save(
        "deduplicated_dialogues.jsonl",
        [dict(record_id=k, **v) for k, v in result["payloads"].items()],
    )
    save(
        "train_candidates.jsonl",
        [result["payloads"][r["source_id"]] for r in result["records"] if r["training_ready"]],
    )
    for bucket, rows in result["previews"].items():
        save("curriculum_cumulative_train_" + bucket + ".jsonl", rows)
    multistep = read_jsonl(config["multistep_inventory"])
    replay = {r["task_id"]: r for r in read_jsonl(config["multistep_replay"])}
    reviews_dir = Path(config["multistep_semantic_reviews"])
    task_ledger = []
    for task in multistep:
        if task["task_id"] not in replay:
            continue
        path = reviews_dir / (task["task_id"] + ".json")
        task_ledger.append(
            dict(
                task,
                reference_replay=replay[task["task_id"]],
                existing_semantic_review=json.loads(path.read_text()) if path.exists() else None,
                existing_semantic_review_sha256=sha256_file(path) if path.exists() else None,
                related_dialogue_record_ids=[
                    r["source_id"]
                    for r in result["records"]
                    if task["task_id"] in r["source_tasks"]
                ],
                complete_training_dialogue=False,
                training_ready=False,
                curriculum_blocker="reference_actions_are_not_complete_accepted_dialogue",
            )
        )
    save("multistep_task_ledger.jsonl", task_ledger)
    for name in (
        "template_overlap",
        "entity_collisions",
        "source_content_variants",
        "manifest",
        "reviewed_training_manifest",
    ):
        (args.output / (name + ".json")).write_text(
            json.dumps(result[name], ensure_ascii=False, indent=2) + "\n"
        )

    def exposure_stats(rows):
        keys = {
            sha256_json(
                dict(messages=r["messages"], message_indices=r["supervision"]["message_indices"])
            )
            for r in rows
        }
        selected = [r for r in result["records"] if r["source_id"] in keys]
        counts = [r["render_token_counts"][-1] for r in selected if r["render_token_counts"]]
        historical = [
            t for t in counts if t["count_basis"] == "historical_identical_messages_and_mask"
        ]
        fresh = [t for t in counts if t["count_basis"] == "fresh_local_tokenizer_render"]
        return dict(
            dialogues=len(selected),
            historically_gradient_exposed=sum(r["historical_gradient_exposure"] for r in selected),
            inherited_token_count_rows=len(historical),
            inherited_assistant_tokens=sum(t["assistant_tokens"] for t in historical),
            inherited_total_tokens=sum(t["total_tokens"] for t in historical),
            fresh_token_count_rows=len(fresh),
            fresh_assistant_tokens=sum(t["assistant_tokens"] for t in fresh),
            fresh_total_tokens=sum(t["total_tokens"] for t in fresh),
            counted_rows=len(counts),
            missing_token_count_rows=len(selected) - len(counts),
            counted_assistant_tokens=sum(t["assistant_tokens"] for t in counts),
            counted_total_tokens=sum(t["total_tokens"] for t in counts),
            token_basis="per_record_historical_or_fresh_provenance; not_all_rows_retokenized",
            illustrative_updates_one_epoch_batch8=(len(selected) + 7) // 8,
            update_count_basis="illustration_only_no_epoch_batch_or_training_run_frozen",
        )

    course_stats = {bucket: exposure_stats(rows) for bucket, rows in result["previews"].items()}
    (args.output / "curriculum_statistics.json").write_text(
        json.dumps(course_stats, ensure_ascii=False, indent=2) + "\n"
    )
    summary = dict(
        version="sft_unified_evidence_v1",
        status="review_inventory_not_training_authorization",
        unique_dialogues=len(result["records"]),
        membership_count=sum(len(r["memberships"]) for r in result["records"]),
        semantic_quality=dict(Counter(r["semantic_quality"] for r in result["records"])),
        source_content_variant_count=len(result["source_content_variants"]),
        template_train_dev_overlap_count=len(result["template_overlap"]),
        entity_train_dev_collision_count=len(result["entity_collisions"]),
        new_review_count=sum(r["new_semantic_review"] is not None for r in result["records"]),
        training_candidate_count=sum(r["training_ready"] for r in result["records"]),
        training_candidate_statistics=exposure_stats(
            [result["payloads"][r["source_id"]] for r in result["records"] if r["training_ready"]]
        ),
        independent_family_dev_ready=False,
        curriculum_cumulative_train_counts={k: len(v) for k, v in result["previews"].items()},
        multistep_task_count=len(task_ledger),
        multistep_complete_accepted_dialogues=0,
        new_composed_pipeline_probe_dialogues=sum(
            r["training_ready"]
            and any(
                m["dataset"] in config.get("new_review_required_cohorts", [])
                for m in r["memberships"]
            )
            for r in result["records"]
        ),
        multistep_scope="original_reference_tasks_remain_unaccepted; authored_composed_probes_counted_separately",
        inputs_sha256={
            **result["inputs_sha256"],
            str(args.ledger_config): sha256_file(args.ledger_config),
            str(args.semantic_reviews): sha256_file(args.semantic_reviews),
        },
        fresh_paid_api_calls=0,
        gpu_hours=0,
        code_sha256=sha256_file(__file__),
        files={p.name: sha256_file(p) for p in args.output.iterdir() if p.is_file()},
    )
    for key in ("multistep_inventory", "multistep_replay"):
        summary["inputs_sha256"][config[key]] = sha256_file(config[key])
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("files", "inputs_sha256")},
            ensure_ascii=False,
        )
    )


def run(args):
    if args.output.exists():
        raise FileExistsError("Never overwrite a reuse inventory")
    report = json.loads((args.distribution / "distribution.json").read_text())
    if sha256_file(args.source) != report["source_sha256"]:
        raise ValueError("Source changed since distribution census")
    args.output.mkdir(parents=True)
    queue = [
        json.loads(lint_item)
        for lint_item in (args.distribution / "source_reuse_queue.jsonl").read_text().splitlines()
    ]
    candidates = {r["source_dialog_id"]: r for r in queue if not r["blocking_flags"]}
    ds, _ = load_complete_airline_dialogues(args.source)
    rows = []
    summaries = []
    for dialogue in ds:
        sid = dialogue.source_dialog_id
        if sid not in candidates:
            continue
        evidence = candidates[sid]
        row = dialogue.to_record()
        row["messages"] = [
            {
                k: v
                for k, v in m.items()
                if k in ("role", "content", "tool_calls", "tool_call_id", "name")
            }
            for m in row["messages"]
        ]
        review_path = args.review / "records" / f"{sid}.json"
        status = "unreviewed_semantic_candidate"
        review_hash = None
        difficulty = None
        if review_path.exists():
            review = json.loads(review_path.read_text())
            review_hash = sha256_file(review_path)
            if review["visible_hash"] != sha256_json(visible_events(row["messages"])):
                status = "stale_review_requires_recheck"
            elif review["status"] != "review_ready":
                status = "prior_review_unresolved"
            else:
                a = review["audit"]
                difficulty = a.get("difficulty")
                if (
                    a["recommendation"] == "keep_candidate"
                    and not a["issues"]
                    and all(
                        r["status"] in ("satisfied", "not_applicable") for r in a["requirements"]
                    )
                ):
                    status = "prior_judge_passed_candidate_not_replay_verified"
                else:
                    status = "prior_semantic_review_hold"
        f = evidence["features"]
        tags = []
        for op in ("flight_or_cabin_change", "cancellation", "passenger_change"):
            if op in f["operation_tags"]:
                tags.append(op)
        if f["tool_calls"] <= 6:
            tags.append("short_observed_trace_not_intrinsic_easy")
        if f["first_tool"] == "get_reservation_details":
            tags.append("reservation_first")
        if "search_onestop_flight" in f["tool_presence"]:
            tags.append("connection_search")
        if "calculate" in f["tool_presence"]:
            tags.append("explicit_calculation_tool")
        row["metadata"].update(
            status=status,
            evidence_level="source_static_and_existing_review_only",
            source_labels=evidence["source_labels"],
            previous_review_sha256=review_hash,
            prior_trajectory_difficulty=difficulty,
            distribution_features=f,
            reuse_priority_tags=tags,
            already_in_train100=evidence["already_in_train100"],
            exact_task_db_mapping=False,
            distribution_inventory_sha256=sha256_file(args.distribution / "distribution.json"),
        )
        row["supervision"] = {
            "version": "approved_assistant_v1",
            "message_indices": evidence["source_target_indices"],
            "basis": "original_source_answer_positions_candidate_not_training_approved",
        }
        rows.append(row)
        summaries.append(
            {
                "source_dialog_id": sid,
                "status": status,
                "already_in_train100": evidence["already_in_train100"],
                "priority_tags": tags,
                "source_target_count": len(evidence["source_target_indices"]),
            }
        )
    # Priority review ordering uses only training-side data; no final IDs/answers.
    statuses = {
        "prior_judge_passed_candidate_not_replay_verified": 0,
        "unreviewed_semantic_candidate": 1,
    }
    rows.sort(
        key=lambda r: (
            statuses.get(r["metadata"]["status"], 2),
            -len(r["metadata"]["reuse_priority_tags"]),
            sha256_json(r["metadata"]["source_dialog_id"]),
        )
    )
    for name, selected in [
        ("all_reuse_candidates", rows),
        (
            "previously_reviewed_candidates",
            [
                r
                for r in rows
                if r["metadata"]["status"] == "prior_judge_passed_candidate_not_replay_verified"
            ],
        ),
    ]:
        (args.output / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected)
        )
    (args.output / "queue.jsonl").write_text("".join(json.dumps(r) + "\n" for r in summaries))
    report = {
        "count": len(rows),
        "statuses": dict(Counter(r["status"] for r in summaries)),
        "new_previously_reviewed_candidates": sum(
            r["status"] == "prior_judge_passed_candidate_not_replay_verified"
            and not r["already_in_train100"]
            for r in summaries
        ),
        "priority_tags": dict(Counter(t for r in summaries for t in r["priority_tags"])),
        "source_sha256": sha256_file(args.source),
        "fresh_paid_calls": 0,
        "training_approved": False,
        "final_task_text_or_answers_used": False,
        "files": {p.name: sha256_file(p) for p in args.output.glob("*.jsonl")},
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("data/raw/areal_tau2/tau2_sft_train.jsonl"))
    p.add_argument("--distribution", type=Path)
    p.add_argument(
        "--review",
        type=Path,
        default=Path("results/analysis/deepseek_rubric_v2_candidates_20260925"),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ledger-config", type=Path)
    p.add_argument("--semantic-reviews", type=Path)
    args = p.parse_args()
    if args.ledger_config:
        if not args.semantic_reviews:
            p.error("--ledger-config requires --semantic-reviews")
        run_ledger(args)
    else:
        if not args.distribution:
            p.error("--distribution is required without --ledger-config")
        run(args)
