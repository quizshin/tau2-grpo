"""Freeze the reviewed 40+10 curriculum; no model calls or source mutations."""

import json
from pathlib import Path

from tau3_grpo.data.manifest import read_manifest
from tau3_grpo.experiments.manifest import build_schedule
from tau3_grpo.utils.hashing import sha256_file, sha256_json

BASE = Path("results/analysis/rl_curriculum_20260912")
SFT = Path("results/analysis/sft4b_data_audit_20260911/airline_sft_train_seed42.jsonl")
OUT = Path("results/analysis/rl_curriculum50_20260912")
REVIEWS = {
    "airline_514": "新预订→加行李→改出发航班→再次加行李；付款与分阶段目标相符。",
    "airline_414": "摄影出差预订→加2件→改早班→再加3件，保留回程；gold最终5件。",
    "airline_296": "家庭预订→加行李→改回程日期→全程升舱；所谓生日纠正与初值相同，是核对而非必须写入。",
    "airline_307": "预订时拼错姓名→两次加行李→纠正姓名，中间询问政策；额外付款方式需在线确认。",
    "airline_658": "拒绝4张礼品卡支付→改用3张加信用卡→预订→改名→加行李→再次改名。",
    "airline_871": "新预订→对3张其他订单逐一加行李→修改另一张订单乘客，保持人数。",
    "airline_184": "拒绝给已飞航段加行李→取消两张重复订单→修改另一张往返订单回程。",
    "airline_1001": "取消3张符合条件的商务舱订单→新建2张不同旅客/行程订单；保留跨订单长流程。",
    "airline_981": "先预订家庭新行程→再取消另一天2张重复订单，保留中间时刻订单；指令明确允许澄清城市。",
    "airline_803": "先升商务舱→再加3件行李，按升级后额度计算1件付费；业务顺序影响费用。",
}
TOOLS = {
    "book_reservation",
    "cancel_reservation",
    "update_reservation_flights",
    "update_reservation_passengers",
    "update_reservation_baggages",
    "send_certificate",
    "transfer_to_human_agents",
}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    core = read_manifest(BASE / "candidate_40_manifest.jsonl")
    original_path = Path("../code_pytrio/data/manifests/areal_airline_train_seed42.jsonl")
    original = read_manifest(original_path)
    audit = json.loads((BASE / "audit.json").read_text())
    assert sha256_file(original_path) == audit["frozen_manifest_hash"]
    assert sha256_file(SFT) == audit["sft_hash"]
    byid = {r.task_id: r for r in original}
    classified = {r["task_id"]: r for r in read_rows(BASE / "all_200_classified.jsonl")}
    sft_seqs = [
        tuple(
            c["name"] for m in d["messages"] for c in m.get("tool_calls", []) if c["name"] in TOOLS
        )
        for d in read_rows(SFT)
    ]
    sft_sets = {frozenset(seq) for seq in sft_seqs}
    clean_tools = {
        tool
        for r in read_rows(BASE / "sft_45_classified.jsonl")
        if r["source_correct"] == 1
        for tool in r["actions"]
    }
    records = []
    for ident, note in REVIEWS.items():
        row = classified[ident]
        assert row["eligible"] and not row["selected"] and not row["gold_replay_errors"]
        entry = byid[ident]
        assert sha256_file(Path("data/raw/areal_tau2") / entry.db_path) == entry.db_hash
        seq = tuple(
            a["name"] for a in entry.task["evaluation_criteria"]["actions"] if a["name"] in TOOLS
        )
        assert len(set(seq)) >= 2 and seq not in sft_seqs and set(seq) <= clean_tools
        records.append(
            dict(
                task_id=ident,
                task_hash=entry.task_hash,
                reason=row["reason"],
                sequence=list(seq),
                novelty="unseen_tool_set"
                if frozenset(seq) not in sft_sets
                else "unseen_exact_sequence",
                review=note,
                limitation="Exact signatures only; not proof of semantic novelty or reward correctness.",
            )
        )
    selected = core + [byid[ident] for ident in REVIEWS]
    ids = [r.task_id for r in selected]
    assert len(ids) == len(set(ids)) == 50 and set(ids) <= set(byid)
    for split in ["selection", "reserve"]:
        excluded = read_manifest(original_path.parent / f"areal_airline_{split}_seed42.jsonl")
        assert not set(ids) & {r.task_id for r in excluded}
    values = [r.model_dump(mode="json") for r in selected]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)

    def dump(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in values)
    (OUT / "manifests/areal_airline_train_seed42.jsonl").write_text(payload)
    dump(
        OUT / "manifests/areal_airline_split_seed42.json",
        dict(
            seed=42,
            split_hash=sha256_json(values),
            parent_split_hash=audit["split_hash"],
            source_revision=selected[0].source_revision,
            train=50,
        ),
    )
    dump(OUT / "new_combinations.json", records)
    schedules = {
        str(n): [
            s.to_dict()
            for s in build_schedule(
                ids, seed=42, group_size=8, groups_per_update=8, total_updates=n
            )
        ]
        for n in [10, 25, 50, 75, 100]
    }
    assert schedules["10"] == schedules["100"][:10]
    dump(OUT / "schedule_preview.json", schedules)
    dump(
        OUT / "audit.json",
        dict(
            core_manifest_hash=sha256_file(BASE / "candidate_40_manifest.jsonl"),
            sft_hash=sha256_file(SFT),
            parent_audit_hash=sha256_file(BASE / "audit.json"),
            manifest_hash=sha256_file(OUT / "manifests/areal_airline_train_seed42.jsonl"),
            script_hash=sha256_file(Path(__file__)),
            core_ids=[r.task_id for r in core],
            new_ids=list(REVIEWS),
            novelty_counts={
                kind: sum(r["novelty"] == kind for r in records)
                for kind in ["unseen_tool_set", "unseen_exact_sequence"]
            },
            trajectories=6400,
            notes="Gold replay reused from prior identical source/DB audit; new combinations reviewed from instructions/actions. No online runs.",
        ),
    )
    print("50 tasks frozen; original 40 preserved; 10 additions; budget 100 x 8 x 8 = 6400")


if __name__ == "__main__":
    main()
