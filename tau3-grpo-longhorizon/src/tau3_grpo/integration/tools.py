"""veRL `BaseTool` implementations backed by the live tau2 Airline Environment.

Milestone D3. Every call goes through `Environment.get_response(ToolCall)` on the
session that owns this rollout, so the DB mutation is real and the recorded
`AssistantMessage` / `ToolCall` / `ToolMessage` triple is exactly what the
official verifier replays. There is no protocol shim and no mock: a tool that
cannot find its session raises rather than inventing a response.

`Environment.get_response` never raises — a failing tool comes back as
`ToolMessage(error=True)` with the error text as content, matching official
behaviour — so this layer does not add its own try/except around it.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from tau3_grpo.env.adapter import airline_tool_schemas
from tau3_grpo.integration.registry import session_for

logger = logging.getLogger(__name__)

#: Airline assistant tools exposed to the policy. Derive this from the live
#: environment so a tau2 snapshot update cannot silently drift from the YAML.
AIRLINE_TOOL_NAMES: tuple[str, ...] = tuple(
    schema["function"]["name"] for schema in airline_tool_schemas()
)


class Tau3AirlineTool(BaseTool):
    """Dispatches one named Airline tool onto the rollout's own Environment."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, str] = {}

    async def create(
        self, instance_id: Optional[str] = None, **kwargs: Any
    ) -> tuple[str, ToolResponse]:
        resolved = instance_id or uuid.uuid4().hex
        self._instances[resolved] = resolved
        return resolved, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs: Any
    ) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        if agent_data is None:
            raise RuntimeError(
                f"tool {self.name} was invoked without agent_data; cannot reach the "
                "tau2 session that owns this rollout"
            )
        entry = session_for(agent_data)
        session = entry.session

        call_id = f"{instance_id}-{session.tool_calls}"
        tool_call = session.make_tool_call(self.name, dict(parameters), call_id)
        session.record_assistant_tool_calls([tool_call])
        tool_message = session.execute_tool_call(tool_call)

        if getattr(tool_message, "error", False):
            entry.tool_error_count += 1

        # The anchor trail is written by the patched ToolAgentLoop through
        # `integration.anchor_hook`, which reads this session after the tool
        # mutates it. Recording it here too would double-count segments.

        metrics = {
            "tool": self.name,
            "error": bool(getattr(tool_message, "error", False)),
            "db_hash": session.db_hash(),
        }
        # Step reward stays 0: reward is terminal and verifiable only.
        return ToolResponse(text=tool_message.content or ""), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        self._instances.pop(instance_id, None)


def build_tool_schema(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Helper for generating the tool YAML consumed by veRL."""

    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }
