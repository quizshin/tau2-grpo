"""AReaL task schema and τ³ conversion boundary.

Milestone D1: parse the real `inclusionAI/AReaL-tau2-data` JSONL format. The
upstream `evaluation_criteria` field is a JSON string and must not be treated as
an already materialized object.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tau3_grpo.utils.hashing import sha256_json


class DataSource(str, Enum):
    AREAL_TAU2_AIRLINE = "areal_tau2_airline"
    TAU3_OFFICIAL_AIRLINE = "tau3_official_airline"


class ArealTaskRecord(BaseModel):
    """One task from AReaL's `tau2_rl_train.jsonl`."""

    model_config = ConfigDict(extra="forbid")

    id: str
    db_path: str
    description: dict[str, Any] | None = None
    user_scenario: dict[str, Any]
    evaluation_criteria: dict[str, Any] | str

    @field_validator("evaluation_criteria")
    @classmethod
    def parse_evaluation_criteria(cls, value: dict[str, Any] | str) -> dict[str, Any]:
        if isinstance(value, str):
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                raise ValueError("evaluation_criteria JSON must decode to an object")
            return parsed
        return value

    @property
    def domain(self) -> str:
        instructions = self.user_scenario.get("instructions", {})
        if isinstance(instructions, dict) and instructions.get("domain"):
            return str(instructions["domain"])
        return "airline" if "airline" in self.db_path.lower() else "unknown"

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.model_dump(mode="json"))

    def resolve_db_path(self, dataset_root: str | Path) -> Path:
        root = Path(dataset_root).resolve()
        candidate = (root / self.db_path).resolve()
        if root not in candidate.parents:
            raise ValueError(f"db_path escapes dataset root: {self.db_path}")
        return candidate

    def to_tau2_task_dict(self) -> dict[str, Any]:
        """Return the shape accepted by τ³-bench v1.0.1 `Task`.

        AReaL actions omit `requestor`; Sierra's model correctly defaults it to
        `assistant`. Reward basis also keeps Sierra's default DB+COMMUNICATE.
        """

        return {
            "id": self.id,
            "description": self.description,
            "user_scenario": self.user_scenario,
            "evaluation_criteria": self.evaluation_criteria,
        }


class OfficialTau3TaskRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_revision: str
    split: str = "base"
    source: DataSource = Field(default=DataSource.TAU3_OFFICIAL_AIRLINE)

