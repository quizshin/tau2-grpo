"""Generate veRL tool YAML from the live tau2-bench Airline schemas."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from tau3_grpo.envs.adapter import airline_tool_schemas
from tau3_grpo.paths import CONFIG_ROOT

LEGACY_SCHEMA = "verl_legacy_v1"
FULL_SCHEMA = "tau3_full_schema_v2"
SCHEMA_VERSIONS = (LEGACY_SCHEMA, FULL_SCHEMA)


def _schema_for_verl(schema: dict) -> dict:
    """Make tau2's OpenAI schema explicit enough for veRL validation."""

    normalized = deepcopy(schema)
    parameters = normalized.get("function", {}).get("parameters")
    if isinstance(parameters, dict):
        # tau2/Pydantic omits ``required`` for a no-argument tool, while veRL's
        # OpenAIFunctionToolSchema requires the field even when it is empty.
        parameters.setdefault("required", [])
    return normalized


def build_config(schema_version: str = LEGACY_SCHEMA) -> dict:
    if schema_version not in SCHEMA_VERSIONS:
        raise ValueError(f"Unknown tool schema version: {schema_version}")
    return {
        "tools": [
            {
                "class_name": "tau3_grpo.envs.tools.Tau3AirlineTool",
                # veRL's registry requires an explicit tool type and dispatches
                # native and MCP tools through different initialization paths.
                "config": {"type": "native", **({"schema_projection": FULL_SCHEMA,
                           "schema_payload": deepcopy(schema)} if schema_version == FULL_SCHEMA else {})},
                "tool_schema": _schema_for_verl(schema),
            }
            for schema in airline_tool_schemas()
        ]
    }


def write_config(path: Path, schema_version: str = LEGACY_SCHEMA) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(build_config(schema_version), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CONFIG_ROOT / "envs/tool_config.yaml")
    parser.add_argument("--schema-version", choices=SCHEMA_VERSIONS, default=LEGACY_SCHEMA)
    args = parser.parse_args(argv)
    path = write_config(args.output, args.schema_version)
    print(f"wrote live tau2 tool config to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
