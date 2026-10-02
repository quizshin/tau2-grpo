"""Offline, prefix-only evidence audit; never edits a dataset or official reward.

Findings deliberately separate contradictions, missing evidence, and semantic review.
Neither a clean static result nor an old judge label certifies a successful trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from tau3_grpo.data.sft_evidence import (
    WRITES as WRITES,
)
from tau3_grpo.data.sft_evidence import (
    audit_record as audit_record,
)
from tau3_grpo.data.sft_evidence import (
    digest as digest,
)
from tau3_grpo.data.sft_evidence import (
    flight_keys as flight_keys,
)
from tau3_grpo.data.sft_evidence import (
    function as function,
)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def run(dataset, output):
    import yaml

    from tau3_grpo.data.sft_expansion import audit_tool_calls
    from tau3_grpo.data.sft_policy_checks import audit_baggage_allowances

    if output.exists():
        raise FileExistsError("Use a new audit directory; preserve prior evidence")
    schemas = [
        t["tool_schema"]
        for t in yaml.safe_load(Path("configs/envs/tool_config.yaml").read_text())["tools"]
    ]
    report, raw, hashes = {}, {}, {}
    for split in ("train", "validation"):
        path = dataset / f"{split}.jsonl"
        raw[split] = rows(path)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        report[split] = []
        for row in raw[split]:
            result = audit_record(row)
            result["schema_findings"] = audit_tool_calls(row, schemas)
            result["baggage_checks"] = audit_baggage_allowances(row["messages"])
            result["disposition"] = (
                "quarantine_for_review"
                if result["schema_findings"]
                or any(f["status"] == "contradiction" for f in result["findings"])
                or any(c["status"] == "violated" for c in result["baggage_checks"])
                else "review_required"
                if result["findings"]
                else "static_checks_clear_semantics_unverified"
            )
            report[split].append(result)
    summary = {}
    for split, items in report.items():
        summary[split] = {
            "count": len(items),
            "dispositions": dict(Counter(r["disposition"] for r in items)),
            "findings": dict(Counter(f["kind"] for r in items for f in r["findings"])),
            "first_tools": dict(Counter(r["first_tool"] for r in items)),
            "legacy_difficulty": dict(Counter(r["historical_difficulty"]["level"] for r in items)),
            "mapped_task_db": sum(r["exact_task_db_mapping_present"] for r in items),
            "matching_stored_execution_receipts": sum(
                r["execution_receipts_match"] is True for r in items
            ),
            "tool_dialogue_coverage": dict(Counter(n for r in items for n in r["tool_counts"])),
            "schema_error_rows": sum(bool(r["schema_findings"]) for r in items),
            "baggage_status": dict(
                Counter(c["status"] for r in items for c in r["baggage_checks"])
            ),
            "payment_checks": dict(Counter(c["status"] for r in items for c in r["checks"])),
        }
    overlaps = {}
    for field in ("source_id", "messages_sha256"):
        overlaps[field] = sorted(
            {r[field] for r in report["train"]} & {r[field] for r in report["validation"]}
        )
    output.mkdir(parents=True)
    for split, items in report.items():
        (output / f"{split}_audit.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items)
        )
    result = dict(
        version="airline_sft_prefix_audit_v1",
        inputs=hashes,
        summary=summary,
        train_dev_overlap=overlaps,
        fresh_environment_replay=False,
        limitations=[
            "Static checks are narrow, not certification of all policies or success.",
            "Legacy difficulty is not a calibrated task-only difficulty label.",
            "No semantic-family independence proof from disjoint source IDs.",
            "No new generation, judge calls, training, or official rescoring.",
        ],
    )
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.output)
