"""Read-only source audit; emit a separate, deterministic RL curriculum candidate.

Run from repository root with PYTHONPATH=. Python needs project data dependencies.
This is static screening, not a semantic verifier or a rollout success predictor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from tau3_grpo.data.manifest import build_airline_splits, load_areal_records, read_manifest
from tau3_grpo.utils.hashing import sha256_file, sha256_json

WRITE = {
    "book_reservation": "booking",
    "cancel_reservation": "cancellation",
    "update_reservation_flights": "flight_change",
    "update_reservation_passengers": "passenger_change",
    "update_reservation_baggages": "baggage",
    "send_certificate": "compensation",
    "transfer_to_human_agents": "transfer",
}
PATTERNS = {
    "booking": r"\bbook (?:a |an |new |my |the |their )|\bnew (?:family |flight|trip)|\bbooking a",
    "cancellation": r"cancel|refund|double.book|duplicate",
    "flight_change": r"reschedule|return (?:leg|flight)|outbound|change (?:the )?(?:date|cabin|flight)|\bmove\b|\bpush\b",
    "passenger_change": r"passenger|\badults\b|children|\bkids\b|\bswap\b|\bsplit\b|reassign|consolidate|overlapping",
    "baggage": r"baggage|\bbags?\b",
    "compensation": r"compensation|compensate",
}


def read_jsonl(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def normalize(s):
    return " ".join(re.findall(r"\w+", s.lower()))


def tags(reason, actions):
    return sorted(
        {k for k, pat in PATTERNS.items() if re.search(pat, reason.lower())}
        | {
            WRITE[a["name"]]
            for a in actions
            if a["name"] in WRITE and a["name"] != "transfer_to_human_agents"
        }
    )


def primary(ts):
    return "multi_goal" if len(ts) > 1 else ts[0] if ts else "lookup_or_policy"


def entities(actions):
    return {
        str(v)
        for a in actions
        for k, v in a.get("arguments", {}).items()
        if k in ("user_id", "reservation_id") and isinstance(v, str)
    }


def mutations(actions):
    return sorted({a["name"] for a in actions if a["name"] in WRITE})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw/areal_tau2/tau2_rl_train.jsonl"))
    ap.add_argument("--frozen-dir", type=Path, default=Path("../code_pytrio/data/manifests"))
    ap.add_argument(
        "--sft-dir", type=Path, default=Path("results/analysis/sft4b_data_audit_20260911")
    )
    ap.add_argument("--out", type=Path, default=Path("results/analysis/rl_curriculum_20260912"))
    ap.add_argument("--size", type=int, default=40)
    ap.add_argument(
        "--review", type=Path, default=Path("scripts/analysis/rl_curriculum_review_20260912.json")
    )
    ap.add_argument(
        "--replay-gold",
        action="store_true",
        help="Replay gold tools on fresh local DBs without any LLM calls",
    )
    ap.add_argument(
        "--reuse-gold-audit",
        type=Path,
        help="Reuse a prior replay with identical source, manifest and DB hashes",
    )
    args = ap.parse_args()
    reviews = json.loads(args.review.read_text())
    if args.replay_gold:
        from loguru import logger

        logger.remove()
        from tau3_grpo.envs.adapter import build_environment, load_flight_db
    sp = build_airline_splits(
        load_areal_records(args.raw), seed=42, source_file_hash=sha256_file(args.raw)
    )
    frozen = read_manifest(args.frozen_dir / "areal_airline_train_seed42.jsonl")
    meta = json.loads((args.frozen_dir / "areal_airline_split_seed42.json").read_text())
    assert meta["split_hash"] == sp.split_hash
    assert [(x.task_id, x.task_hash) for x in frozen] == [
        (x.task_id, x.task_hash) for x in sp.train
    ]
    replay_cache = None
    if args.reuse_gold_audit:
        audit = json.loads((args.reuse_gold_audit / "audit.json").read_text())
        assert audit["gold_replayed"]
        assert audit["source_hash"] == sha256_file(args.raw)
        assert audit["frozen_manifest_hash"] == sha256_file(
            args.frozen_dir / "areal_airline_train_seed42.jsonl"
        )
        assert all(
            sha256_file(args.raw.parent / name) == digest
            for name, digest in audit["db_hashes"].items()
        )
        replay_cache = {
            r["task_id"]: r["gold_replay_errors"]
            for r in read_jsonl(args.reuse_gold_audit / "all_200_classified.jsonl")
        }
        assert set(replay_cache) == {x.task_id for x in frozen}
    sf = args.sft_dir / "airline_sft_train_seed42.jsonl"
    labels = json.loads((args.sft_dir / "source_labels.json").read_text())["metadata_by_id"]
    sft = []
    for x in read_jsonl(sf):
        m = x["metadata"]
        ident = m["source_dialog_id"]
        acts = [c for msg in x["messages"] for c in msg.get("tool_calls", [])]
        ts = tags(m["reason_for_call"], acts)
        sft.append(
            dict(
                id=ident,
                reason=m["reason_for_call"],
                tags=ts,
                primary=primary(ts),
                actions=mutations(acts),
                entities=sorted(entities(acts)),
                source_correct=labels.get(ident, {}).get("correct"),
                seed_pattern=labels.get(ident, {}).get("seed_pattern_task_id"),
                scenario_id=m.get("scenario_id"),
            )
        )
    clean = [x for x in sft if x["source_correct"] == 1]
    rows = []
    dbs = {}
    for entry in frozen:
        task = entry.task
        ins = task["user_scenario"]["instructions"]
        ev = task["evaluation_criteria"]
        acts = ev.get("actions") or []
        reason = ins.get("reason_for_call", "")
        ts = tags(reason, acts)
        mut = mutations(acts)
        ent = entities(acts)
        path = args.raw.parent / entry.db_path
        issues = []
        if not path.exists():
            issues.append("missing_db")
        else:
            if entry.db_path not in dbs:
                dbs[entry.db_path] = (sha256_file(path), json.loads(path.read_text()))
            dh, db = dbs[entry.db_path]
            if dh != entry.db_hash:
                issues.append("db_hash_mismatch")
            for a in acts:
                for k, table in [("user_id", "users"), ("reservation_id", "reservations")]:
                    val = a.get("arguments", {}).get(k)
                    # IDs created during booking may legitimately be absent initially.
                    if (
                        val
                        and val not in db[table]
                        and not any(z["name"] == "book_reservation" for z in acts)
                    ):
                        issues.append("unresolved_initial_" + k + ":" + str(val))
        if (
            not any(a["name"] in WRITE for a in acts)
            and not ev.get("communicate_info")
            and not ev.get("nl_assertions")
        ):
            issues.append("read_only_gold_without_communication_checks")
        if not any(ev.get(k) for k in ["actions", "communicate_info", "nl_assertions"]):
            issues.append("empty_evaluation")
        replay_errors = []
        if args.replay_gold and path.exists():
            env = build_environment(load_flight_db(path))
            for index, action in enumerate(acts):
                try:
                    result = env.use_tool(action["name"], **action["arguments"])
                    if isinstance(result, dict) and result.get("error"):
                        raise ValueError(str(result["error"]))
                    if isinstance(result, str) and result.lower().startswith("error"):
                        raise ValueError(result)
                except Exception as exc:
                    replay_errors.append(
                        dict(index=index, tool=action["name"], error=str(exc)[:600])
                    )
                    break  # Later calls may depend on this failed state transition.
        if replay_cache is not None and not args.replay_gold:
            replay_errors = replay_cache[entry.task_id]
        review = reviews.get(entry.task_id)
        if replay_errors:
            issues.append("gold_replay_error")
        if review and review["status"] == "quarantine":
            issues.append("review_goal_reward_conflict")

        def match(s):
            overlap = set(ts) & set(s["tags"])
            union = set(ts) | set(s["tags"])
            score = len(overlap) / max(1, len(union))
            lex = SequenceMatcher(None, normalize(reason), normalize(s["reason"])).ratio()
            shared = sorted(ent & set(s["entities"]))
            return dict(
                sft_id=s["id"],
                capability_jaccard=round(score, 4),
                lexical_similarity=round(lex, 4),
                exact_reason=normalize(reason) == normalize(s["reason"]),
                shared_entities=shared,
                same_mutation_set=mut == s["actions"],
                source_correct=s["source_correct"],
            )

        matches = sorted(
            [match(s) for s in sft],
            key=lambda m: (
                m["exact_reason"],
                m["capability_jaccard"],
                m["same_mutation_set"],
                m["lexical_similarity"],
            ),
            reverse=True,
        )
        cm = [m for m in matches if m["source_correct"] == 1]
        best = cm[0]
        support = sorted({a for s in clean for a in s["actions"]})
        supported = set(mut).issubset(support)
        relation = (
            "same_reason_and_entities_candidate"
            if best["exact_reason"] and best["shared_entities"]
            else "same_reason_candidate"
            if best["exact_reason"]
            else "same_capability_combination"
            if best["capability_jaccard"] == 1 and ts
            else "partial_capability_overlap"
            if best["capability_jaccard"] > 0
            else "weak_or_no_link"
        )
        rows.append(
            dict(
                task_id=entry.task_id,
                reason=reason,
                tags=ts,
                primary=primary(ts),
                gold_action_count=len(acts),
                mutation_tools=mut,
                entities=sorted(ent),
                db_path=entry.db_path,
                complexity="long" if len(acts) > 10 else "medium" if len(acts) > 5 else "short",
                relation=relation,
                best_clean_match=best,
                top_matches=matches[:3],
                unsupported_mutation_tools=sorted(set(mut) - set(support)),
                structural_issues=sorted(set(issues)),
                review=review,
                gold_replay_errors=replay_errors,
                policy_negative_hint=bool(
                    re.search(
                        r"cannot|ineligible|refus|deni|not allowed",
                        ins.get("task_instructions", "").lower(),
                    )
                ),
                eligible=not issues and supported and best["capability_jaccard"] > 0,
            )
        )
    # Cover every primary category, then fill underrepresented category/length strata.
    # Prefer SFT linkage within strata, while discouraging repeated entity templates.
    chosen = []
    pool = [r for r in rows if r["eligible"]]
    assert 0 < args.size <= len(pool)
    while len(chosen) < args.size:
        counts = Counter(r["primary"] for r in chosen)
        strata = Counter((r["primary"], r["complexity"]) for r in chosen)
        users = Counter(e for r in chosen for e in r["entities"])

        def rank(r):
            return (
                counts[r["primary"]],
                strata[(r["primary"], r["complexity"])],
                sum(users[e] for e in r["entities"]),
                -r["best_clean_match"]["capability_jaccard"],
                -r["best_clean_match"]["lexical_similarity"],
                hashlib.sha256(("42:" + r["task_id"]).encode()).hexdigest(),
            )

        r = min(pool, key=rank)
        chosen.append(r)
        pool.remove(r)
    chosen_ids = {r["task_id"] for r in chosen}
    assert len(chosen_ids) == args.size
    assert chosen_ids <= {x.task_id for x in sp.train}
    assert not chosen_ids & {x.task_id for x in sp.selection + sp.reserve}
    for r in rows:
        r["selected"] = r["task_id"] in chosen_ids
        r["decision"] = (
            "selected_stratified_candidate"
            if r["selected"]
            else "quarantine_structural_issue"
            if r["structural_issues"]
            else "defer_weak_support"
            if not r["eligible"]
            else "defer_budget_or_duplicate_coverage"
        )
    args.out.mkdir(parents=True, exist_ok=True)

    def dump(name, obj):
        (args.out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")

    def jsonl(name, records):
        (args.out / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
        )

    jsonl("all_200_classified.jsonl", rows)
    jsonl("sft_45_classified.jsonl", sft)
    byid = {x.task_id: x for x in frozen}
    jsonl(
        f"candidate_{args.size}_manifest.jsonl",
        [byid[r["task_id"]].model_dump(mode="json") for r in chosen],
    )
    summary = dict(
        source_hash=sha256_file(args.raw),
        split_hash=sp.split_hash,
        sft_hash=sha256_file(sf),
        frozen_manifest_hash=sha256_file(args.frozen_dir / "areal_airline_train_seed42.jsonl"),
        labels_hash=sha256_file(args.sft_dir / "source_labels.json"),
        script_hash=sha256_file(Path(__file__)),
        review_hash=sha256_file(args.review),
        gold_replayed=args.replay_gold or replay_cache is not None,
        replay_reused_from=str(args.reuse_gold_audit) if args.reuse_gold_audit else None,
        db_hashes={k: v[0] for k, v in dbs.items()},
        counts=dict(
            sft=len(sft),
            sft_label_clean=len(clean),
            rl=len(rows),
            eligible=sum(r["eligible"] for r in rows),
            selected=len(chosen),
        ),
        sft_primary=dict(Counter(r["primary"] for r in sft)),
        rl_primary=dict(Counter(r["primary"] for r in rows)),
        selected_primary=dict(Counter(r["primary"] for r in chosen)),
        rl_relations=dict(Counter(r["relation"] for r in rows)),
        selected_complexity=dict(Counter(r["complexity"] for r in chosen)),
        structural_issue_tasks=[r["task_id"] for r in rows if r["structural_issues"]],
        issue_counts=dict(Counter(i for r in rows for i in r["structural_issues"])),
        selected_policy_negative_hints=sum(r["policy_negative_hint"] for r in chosen),
        selected_ids=[r["task_id"] for r in chosen],
        caveats=[
            "Static heuristic labels require semantic review; CPU tool replay does not verify policy, rewards or solvability.",
            "SFT correct=1 is a source label, not independent proof.",
            "No exact task identity established; scenario_id missing.",
            "New-off checkpoint training snapshot has not been independently tied to this SFT hash.",
            "Selection/reserve excluded by frozen split. No new semantic dedup against official final performed.",
            "Gold action count is a complexity proxy, not measured dialogue length.",
        ],
    )
    dump("audit.json", summary)
    manifest_dir = args.out / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    selected_records = [byid[r["task_id"]].model_dump(mode="json") for r in chosen]
    (manifest_dir / "areal_airline_train_seed42.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in selected_records)
    )
    (manifest_dir / "areal_airline_split_seed42.json").write_text(
        json.dumps(
            dict(
                seed=42,
                split_hash=sha256_json(selected_records),
                parent_split_hash=sp.split_hash,
                source_revision=sp.source_revision,
                train=len(chosen),
                purpose="screened_curriculum_candidate",
                selection_source="unchanged original 60-task selection split",
            ),
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# RL 缩减候选：200 → 40",
        "",
        "本报告只筛查数据；未修改正式配置或启动训练。候选不是已验证可解的任务集。",
        "",
        "## 任务分类（目标关键词与动作标签的并集；多目标单独计数）",
        "",
        "|主类|SFT 45|原 RL 200|候选 40|",
        "|---|---:|---:|---:|",
    ]
    for cat in sorted(set(summary["sft_primary"]) | set(summary["rl_primary"])):
        lines.append(
            f"|{cat}|{summary['sft_primary'].get(cat, 0)}|{summary['rl_primary'].get(cat, 0)}|{summary['selected_primary'].get(cat, 0)}|"
        )
    lines += [
        "",
        "## 衔接证据与限制",
        "",
        json.dumps(summary["rl_relations"], ensure_ascii=False),
        "",
        f"结构问题任务：{summary['structural_issue_tasks']}；可参与候选分层：{summary['counts']['eligible']}。",
        "",
        "SFT 的 4 条 correct=0 示范保留在审计中，但不作为干净支持依据。correct=1 也不代表人工复核通过。",
        "动作类型覆盖、目标文本相同、实体重合均不等于同一任务；缺少 scenario_id，无法证明精确任务继承。",
        "先平衡主类，再平衡参考动作长度，优先避免重复实体模板并选取与 SFT 能力相近者；不是按成功率挑容易任务。",
        "短/中/长按参考动作数 ≤5 / 6–10 / >10，仅作静态复杂度代理。",
        "冻结 200 名单已用原始数据与项目 fingerprint 算法重建，逐项核对 ID/hash、split_hash，核对候选 DB hash。",
        "候选只来自原 train，selection60/reserve888 完全隔离；未另做官方 final 的语义去重。",
        "当前 new-off 模型的数据快照尚未与此 45 条哈希独立核对，因此这是对已审计 SFT 快照的衔接筛查。",
        "",
        "CPU gold 工具回放："
        + str(summary["gold_replayed"])
        + "；问题类型计数："
        + json.dumps(summary["issue_counts"], ensure_ascii=False),
        "逐项问题、人工阅读备注及工具报错均见 all_200_classified.jsonl。只读 gold 且无沟通检查的任务暂缓，需加强验证器。",
        "",
        "## 候选任务（每项完整输入/评分动作见 manifest）",
        "",
        "|任务|主类|参考动作数|相近的干净标签 SFT|目标|",
        "|---|---|---:|---|---|",
    ]
    for r in chosen:
        lines.append(
            f"|{r['task_id']}|{r['primary']}|{r['gold_action_count']}|{r['best_clean_match']['sft_id']}|{r['reason'].replace('|', '/')}|"
        )
    lines += [
        "",
        "## 建议训练预算",
        "",
        "40 个固定训练任务；每批 8 个任务 × 每任务 8 次采样 = 64 条在线轨迹。",
        "先累计 10 批（640 条）；通过轨迹/奖励检查后，第一阶段累计上限 50 批（3200 条）。",
        "40 任务均衡轮转时，10 批平均每任务出现 2 次（16 条轨迹）；50 批出现 10 次（80 条轨迹）。",
        "这里的批指在线采样/更新批，不能把 PPO minibatch 再乘入轨迹预算。E1/E3 屏蔽全 0/全 1 组的梯度，不额外补采；有效组更少，但候选轨迹预算不变。",
        "正式上线前：复核候选的用户指令与 gold 动作一致性、策略可行性与奖励；核对 new-off 的 SFT 数据身份。",
        "保留原 60 条 selection 和官方 final 作独立评测；不能依据 final 结果反选训练任务。",
    ]
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("db_hashes", "selected_ids")},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
