"""Explicit, hash-bound SFT prompt preservation; shared RL prompts stay unchanged."""

import json
import re
from copy import deepcopy

from tau3_grpo.prompts import build_system_prompt, prepare_agent_messages, prompt_provenance
from tau3_grpo.utils.hashing import sha256_json, sha256_text


def prepare_sft_messages(messages, frozen_prompt_protocols=None):
    if frozen_prompt_protocols is None:
        return prepare_agent_messages(messages), prompt_provenance()
    if not isinstance(frozen_prompt_protocols, dict) or not frozen_prompt_protocols:
        raise ValueError("Frozen SFT prompts require a nonempty hash-to-protocol mapping")
    if not messages or messages[0].get("role") != "system":
        raise ValueError("Frozen SFT dialogue requires a source system prompt")
    prompt = messages[0].get("content")
    if not isinstance(prompt, str):
        raise ValueError("Frozen SFT system content must be text")
    digest = sha256_text(prompt)
    version = re.search(r'<tool_call_protocol version="([^"]+)">', prompt)
    if (digest not in frozen_prompt_protocols or version is None
            or frozen_prompt_protocols[digest] != version.group(1)
            or version.group(1) not in {
                "airline_sequential_multicall_v1", "airline_teacher_visible_text_multicall_v2"}):
        raise ValueError("Source prompt does not match the frozen SFT prompt manifest")
    marker = "<policy>\n"
    canonical = build_system_prompt()
    if prompt.count(marker) != 1 or prompt.split(marker, 1)[1] != canonical.split(marker, 1)[1]:
        raise ValueError("Frozen SFT prompt changed the pinned airline business policy")
    prepared = deepcopy(messages)
    converted = []
    for mi, message in enumerate(prepared):
        for ci, call in enumerate(message.get("tool_calls", [])):
            function = call.get("function", call)
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                # OpenAI wire format uses JSON strings; Qwen's native template
                # requires objects. Parse exactly, never repair invalid JSON.
                arguments = json.loads(arguments)
                converted.append([mi, ci])
            if not isinstance(arguments, dict):
                raise ValueError("Frozen SFT tool arguments must decode to an object")
            function["arguments"] = arguments
    return prepared, {
        **prompt_provenance(prompt),
        "tool_protocol": version.group(1),
        "sft_prompt_mode": "frozen_source_v1",
        "source_messages_sha256": sha256_json(messages),
        "render_messages_sha256": sha256_json(prepared),
        "tool_argument_json_decodes": converted,
    }
