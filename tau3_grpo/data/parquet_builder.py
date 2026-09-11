"""veRL parquet builder.

Milestone D8. One row per training task with the real `interaction_kwargs` the
`Tau3AirlineInteraction` needs, and a system prompt that carries the official tau2
Airline business rules with the project multi-call protocol. The builder refuses τ³ official rows, so the training
parquet cannot contain final-set tasks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from tau3_grpo.data.manifest import ManifestEntry
from tau3_grpo.data.official import assert_trainable_entries
from tau3_grpo.paths import PARQUET_ROOT
from tau3_grpo.prompts import build_system_prompt, prompt_provenance

#: Must match `tau3_grpo.envs.interaction.INTERACTION_NAME`. Duplicated as a
#: literal so building a parquet does not require veRL to be installed; the
#: interaction module asserts the two agree.
INTERACTION_NAME = "tau3_airline"

def build_row(
    entry: ManifestEntry,
    *,
    policy: str,
    split: str,
    anchor_mode: str = "structured",
    anchor_similarity_threshold: float = 0.9,
    seed: Optional[int] = None,
    tools_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build one veRL dataset row for a manifest entry."""

    task = entry.task or {}
    instructions = task.get("user_scenario", {}).get("instructions", {})
    opening = ""
    if isinstance(instructions, dict):
        opening = str(instructions.get("reason_for_call", "") or "")

    interaction_kwargs = {
        "name": INTERACTION_NAME,
        "task_id": entry.task_id,
        "source": entry.source.value,
        "split": split,
        # The interaction object is constructed once inside each rollout
        # worker, so it cannot receive a driver-process SessionFactory. Carry
        # the pinned record in the row and adapt it lazily per trajectory.
        # Keep the heterogeneous AReaL record out of Arrow's nested-struct
        # inference. Some records contain empty dicts or variant fields, which
        # PyArrow cannot encode as one stable struct schema.
        "record_json": json.dumps(task, sort_keys=True, separators=(",", ":")),
        # The first user message is already present in the policy prompt.  Carry
        # the same text into the tau2 session so the user simulator history,
        # official replay and the t=1 anchor all observe identical state.
        "initial_user_message": opening
        or "Hello, I need help with my reservation.",
        "anchor_mode": anchor_mode,
        "anchor_similarity_threshold": anchor_similarity_threshold,
    }
    if seed is not None:
        interaction_kwargs["seed"] = seed

    extra_info = {
        **prompt_provenance(build_system_prompt(policy)),
        "index": entry.task_id,
        "task_id": entry.task_id,
        "split": split,
        "source": entry.source.value,
        "source_revision": entry.source_revision,
        "task_hash": entry.task_hash,
        "db_path": entry.db_path,
        "db_hash": entry.db_hash,
        "need_tools_kwargs": bool(tools_kwargs),
        "interaction_kwargs": interaction_kwargs,
    }
    if tools_kwargs:
        extra_info["tools_kwargs"] = tools_kwargs

    return {
        "data_source": entry.source.value,
        "prompt": [
            {"role": "system", "content": build_system_prompt(policy)},
            {"role": "user", "content": opening or "Hello, I need help with my reservation."},
        ],
        "ability": "airline_customer_service",
        "reward_model": {"style": "rule", "ground_truth": entry.task_hash},
        "extra_info": extra_info,
    }


def build_rows(
    entries: Sequence[ManifestEntry],
    *,
    policy: str,
    split: str,
    anchor_mode: str = "structured",
    anchor_similarity_threshold: float = 0.9,
    seed: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Build every row, rejecting τ³ official entries first."""

    assert_trainable_entries(list(entries), context="parquet_builder.build_rows")
    return [
        build_row(
            entry,
            policy=policy,
            split=split,
            anchor_mode=anchor_mode,
            anchor_similarity_threshold=anchor_similarity_threshold,
            seed=seed,
        )
        for entry in entries
    ]


def write_parquet(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
) -> Path:
    """Write rows to parquet via pandas/pyarrow."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pandas is required to write parquet; install the 'data' extra") from exc

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_parquet(target, index=False)
    return target


def build_and_write(
    entries: Sequence[ManifestEntry],
    *,
    policy: str,
    split: str,
    output_dir: str | Path = PARQUET_ROOT,
    filename: Optional[str] = None,
    anchor_mode: str = "structured",
    seed: Optional[int] = None,
) -> Path:
    """Build rows for a split and write them to parquet."""

    rows = build_rows(
        entries, policy=policy, split=split, anchor_mode=anchor_mode, seed=seed
    )
    name = filename or f"airline_{split}.parquet"
    return write_parquet(rows, Path(output_dir) / name)
