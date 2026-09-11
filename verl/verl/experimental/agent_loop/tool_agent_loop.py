# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import importlib
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# --- Tau3-GRPO local patch: anchor id / token span emission ------------------
# The Tau-GiGPO estimator needs to know which response tokens belong to which
# assistant generation, and which anchor (environment state) that generation was
# conditioned on. veRL itself has no notion of an anchor, so the project registers
# a callable here instead of veRL importing the project.
#
# The hook receives (agent_data, segment_kind) and returns an anchor id or None.
# It is called once per appended token segment, so `anchor_ids` and `anchor_spans`
# stay index-aligned with the token stream: real ids at assistant segments, None at
# tool and user observation segments.
_TAU3_ANCHOR_HOOK = None


def _load_tau3_anchor_hook_from_env():
    """Tau3-GRPO local patch: lazily load the hook inside each rollout worker."""

    global _TAU3_ANCHOR_HOOK
    if _TAU3_ANCHOR_HOOK is not None:
        return _TAU3_ANCHOR_HOOK
    spec = os.getenv("TAU3_GRPO_ANCHOR_HOOK", "")
    if not spec:
        return None
    module_name, separator, object_name = spec.partition(":")
    if not separator:
        raise ValueError("TAU3_GRPO_ANCHOR_HOOK must use 'module:callable' syntax")
    _TAU3_ANCHOR_HOOK = getattr(importlib.import_module(module_name), object_name)
    return _TAU3_ANCHOR_HOOK


def set_tau3_anchor_hook(hook):
    """Tau3-GRPO local patch: register the anchor resolver (None disables it)."""

    global _TAU3_ANCHOR_HOOK
    _TAU3_ANCHOR_HOOK = hook


def get_tau3_anchor_hook():
    """Tau3-GRPO local patch: return the registered anchor resolver."""

    return _TAU3_ANCHOR_HOOK


def tau3_anchor_hook(agent_data, segment_kind, start, end):
    """Tau3-GRPO local patch: record one token segment's anchor id and span.

    `segment_kind` is "assistant", "tool" or "user". Observation segments always
    record None so the per-trajectory lists line up with the token spans.
    """

    anchor_id = None
    hook = _TAU3_ANCHOR_HOOK
    if segment_kind == "assistant" and hook is None:
        try:
            hook = _load_tau3_anchor_hook_from_env()
        except Exception as exc:  # pragma: no cover - fail closed to no step credit
            logger.warning(f"tau3 anchor hook import failed: {exc}")
    if segment_kind == "assistant" and hook is not None:
        try:
            anchor_id = hook(agent_data, segment_kind)
        except Exception as exc:  # pragma: no cover - never break a rollout
            logger.warning(f"tau3 anchor hook failed: {exc}")
            anchor_id = None
    agent_data.anchor_ids.append(anchor_id)
    agent_data.anchor_spans.append((int(start), int(end)) if anchor_id is not None else None)


# --- end Tau3-GRPO local patch ----------------------------------------------


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.termination_reason: Optional[str] = None
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []
        # Parser-separated prose for the official tool-call replay history.
        self.assistant_content: Optional[str] = None

        # Tau3-GRPO local patch: per-segment anchor ids and token spans.
        self.anchor_ids: list[Optional[str]] = []
        self.anchor_spans: list[Optional[tuple[int, int]]] = []

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize tools from config file
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.tool_execution_mode = self.rollout_config.multi_turn.tool_execution_mode
        if self.tool_execution_mode not in {"sequential", "parallel"}:
            raise ValueError("tool_execution_mode must be sequential or parallel")
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # Initialize interactions from config file
        self.interaction_config_file = self.rollout_config.multi_turn.interaction_config_path
        if self.interaction_config_file:
            self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(
                self.interaction_config_file
            )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

        # State machine loop. Tau3-GRPO local patch: release the private tau2
        # session even when generation, a tool, or the simulator raises.
        state = AgentState.PENDING
        try:
            while state != AgentState.TERMINATED:
                if state == AgentState.PENDING:
                    state = await self._handle_pending_state(agent_data, sampling_params)
                elif state == AgentState.GENERATING:
                    state = await self._handle_generating_state(agent_data, sampling_params)
                elif state == AgentState.PROCESSING_TOOLS:
                    state = await self._handle_processing_tools_state(agent_data)
                elif state == AgentState.INTERACTING:
                    state = await self._handle_interacting_state(agent_data)
                else:
                    logger.error(f"Invalid state: {state}")
                    agent_data.termination_reason = "unexpected_error"
                    state = AgentState.TERMINATED
        except BaseException:
            if agent_data.interaction is not None:
                try:
                    await agent_data.interaction.finalize_interaction(agent_data.request_id)
                except Exception as cleanup_exc:  # pragma: no cover - preserve original error
                    logger.warning(f"interaction cleanup failed: {cleanup_exc}")
            raise

        # Tau3-GRPO local patch: terminal verifier reward is computed inside the
        # rollout worker while its private tau2 session still exists. Publishing
        # it as AgentLoopOutput.reward_score makes veRL create rm_scores directly.
        terminal_reward_score = None
        if agent_data.interaction is not None:
            finalizer = getattr(agent_data.interaction, "finalize_rollout", None)
            if callable(finalizer):
                terminal_payload = await finalizer(
                    agent_data.request_id,
                    termination_reason=agent_data.termination_reason or "agent_stop",
                    anchor_ids=agent_data.anchor_ids,
                    anchor_spans=agent_data.anchor_spans,
                )
                terminal_reward_score = float(terminal_payload["reward"])
                # `_postprocess` converts each reward-extra value with
                # `np.array`. Keep nested verifier payloads JSON encoded so
                # variable-length reward bases/trajectories cannot form ragged
                # arrays and crash a mixed rollout batch.
                scalar_keys = (
                    "task_id",
                    "termination_reason",
                    "failure_category",
                    "db_hash",
                    "initial_db_hash",
                    "scored",
                    "tool_protocol",
                    "agent_system_prompt_sha256",
                    "benchmark_policy_modified",
                )
                reward_extra_info = {
                    key: terminal_payload.get(key) for key in scalar_keys
                }
                for source_key, output_key in (
                    ("reward_breakdown", "reward_breakdown_json"),
                    ("reward_basis", "reward_basis_json"),
                    ("info", "verifier_info_json"),
                    ("trajectory", "trajectory_json"),
                ):
                    reward_extra_info[output_key] = json.dumps(
                        terminal_payload.get(source_key), sort_keys=True
                    )
                agent_data.extra_fields["reward_extra_info"] = reward_extra_info
            else:
                await agent_data.interaction.finalize_interaction(agent_data.request_id)

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            reward_score=terminal_reward_score,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        # Tau3-GRPO local patch: publish the anchor trail. `_postprocess` turns every
        # extra_fields key into a non_tensor_batch key, so the tau_gigpo estimator can
        # read anchor_ids / anchor_spans without further plumbing.
        output.extra_fields.update(
            {"anchor_ids": agent_data.anchor_ids, "anchor_spans": agent_data.anchor_spans}
        )
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        # Tau3-GRPO local patch: old parquet files carry the old system prompt.
        # Adapt it before tokenization, so the actor trains on the served prompt.
        prepare_messages = getattr(agent_data.interaction, "prepare_agent_messages", None)
        if callable(prepare_messages):
            required_mode = getattr(agent_data.interaction, "required_tool_execution_mode", None)
            if required_mode and self.tool_execution_mode != required_mode:
                raise ValueError(f"this interaction requires tool_execution_mode={required_mode}")
            agent_data.messages = prepare_messages(agent_data.messages)
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=self.tool_schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        if len(prompt_ids) > self.prompt_length:
            raise ValueError("prepared agent prompt exceeds prompt_length")
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        # Tau3-GRPO local patch: cap the *next* generation, not the response
        # just produced. The last allowed turn must reach its tool/interaction
        # handler and be recorded before the trajectory can be finalized.
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            agent_data.termination_reason = "max_steps"
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            agent_data.termination_reason = "max_steps"
            return AgentState.TERMINATED

        # Tau3-GRPO local patch: the v2-2 contract caps *each assistant turn* at
        # 1,024 new tokens while keeping a much larger trajectory budget.  veRL's
        # default async server otherwise gives every turn the whole remaining
        # response budget, which can let one malformed turn consume the complete
        # 24k context.  Copy because sampling_params is shared by concurrent loops
        # and the vLLM adapter pops max_tokens.
        remaining_tokens = self.response_length - len(agent_data.response_mask)
        if remaining_tokens <= 0:
            agent_data.termination_reason = "context_window_exceeded"
            return AgentState.TERMINATED
        max_tokens_per_turn = int(os.getenv("TAU3_GRPO_MAX_TOKENS_PER_TURN", "1024"))
        if max_tokens_per_turn <= 0:
            raise ValueError("TAU3_GRPO_MAX_TOKENS_PER_TURN must be positive")
        turn_sampling_params = dict(sampling_params)
        turn_sampling_params["max_tokens"] = min(max_tokens_per_turn, remaining_tokens)

        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=turn_sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            # Multi-round calls, only update the maximum max_global_steps.
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        # Tau3-GRPO local patch: span of this assistant generation inside the response.
        _tau3_span_start = len(agent_data.response_mask)
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        tau3_anchor_hook(agent_data, "assistant", _tau3_span_start, len(agent_data.response_mask))
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            agent_data.termination_reason = "context_window_exceeded"
            return AgentState.TERMINATED
        # Extract tool calls
        tools = [tool.tool_schema for tool in self.tools.values()]
        agent_data.assistant_content, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(
            agent_data.response_ids, tools
        )

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            agent_data.termination_reason = "agent_stop"
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        # Tau3-GRPO local patch: ordered full-list execution is opt-in. Keep the
        # upstream mode available for unrelated veRL users of this vendored tree.
        sequential = self.tool_execution_mode == "sequential"
        calls = agent_data.tool_calls if sequential else agent_data.tool_calls[: self.max_parallel_calls]
        recorded_calls = [None] * len(calls)
        recorder = getattr(agent_data.interaction, "record_tool_batch", None)
        if sequential and callable(recorder):
            recorded_calls = recorder(
                agent_data.request_id, tool_calls=calls,
                assistant_content=agent_data.assistant_content,
            )
            if len(recorded_calls) != len(calls):
                raise RuntimeError("tool batch recording changed the number of calls")

        tool_call_names = [call.name for call in calls]
        with simple_timer("tool_calls", agent_data.metrics):
            if sequential:
                responses = []
                for call, recorded_call in zip(calls, recorded_calls, strict=True):
                    responses.append(await self._call_tool(
                        call, agent_data.tools_kwargs, agent_data,
                        recorded_tool_call=recorded_call,
                    ))
            else:
                responses = await asyncio.gather(*[
                    self._call_tool(call, agent_data.tools_kwargs, agent_data) for call in calls
                ])

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for (tool_response, tool_reward, _), recorded_call in zip(responses, recorded_calls, strict=True):
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            if recorded_call is not None:
                message["tool_call_id"] = recorded_call.id
                message["name"] = recorded_call.name
            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            agent_data.termination_reason = "context_window_exceeded"
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        # Tau3-GRPO local patch: tool observation tokens carry no anchor.
        _tau3_span_start = len(agent_data.response_mask)
        agent_data.response_mask += [0] * len(response_ids)
        tau3_anchor_hook(agent_data, "tool", _tau3_span_start, len(agent_data.response_mask))
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        # Tau3-GRPO local patch: user observation tokens carry no anchor.
        _tau3_span_start = len(agent_data.response_mask)
        agent_data.response_mask += [0] * len(response_ids)
        tau3_anchor_hook(agent_data, "user", _tau3_span_start, len(agent_data.response_mask))
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            agent_data.termination_reason = metrics.get("termination_reason", "user_stop")
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData,
        *, recorded_tool_call: Any = None,
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        tool_name = str(getattr(tool_call, "name", "unknown_tool"))
        raw_arguments = getattr(tool_call, "arguments", "")
        dispatch_ready = False
        try:
            tool_args = json.loads(raw_arguments)
            if not isinstance(tool_args, dict):
                raise TypeError(
                    f"tool arguments must decode to an object, got {type(tool_args).__name__}"
                )
            tool = self.tools[tool_name]
            dispatch_ready = True
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            execution_kwargs = {"agent_data": agent_data}
            if recorded_tool_call is not None:
                execution_kwargs["recorded_tool_call"] = recorded_tool_call
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, **execution_kwargs
            )
        except Exception as e:
            # Official tools return ordinary errors as ToolMessages. An actual
            # exception after dispatch begins is an infrastructure/wrapper
            # failure with potentially applied writes: do not retry or score it.
            if recorded_tool_call is not None and dispatch_ready:
                raise
            logger.warning(f"Error when executing tool: {e}")
            error_text = f"Error when executing tool: {e}"
            # Tau3-GRPO local patch: parser and dispatch errors happen before
            # Tau3AirlineTool can write the AssistantMessage/ToolMessage pair.
            # Let the interaction record the failed call through the live tau2
            # Environment so the official verifier replays the same no-op.
            if agent_data.interaction is not None:
                recorder = getattr(agent_data.interaction, "record_tool_failure", None)
                if callable(recorder):
                    try:
                        recording_kwargs = {}
                        if recorded_tool_call is not None:
                            recording_kwargs["recorded_tool_call"] = recorded_tool_call
                        error_text = recorder(
                            agent_data.request_id,
                            tool_name=tool_name,
                            raw_arguments=raw_arguments,
                            error=str(e),
                            assistant_content=agent_data.assistant_content,
                            **recording_kwargs,
                        )
                    except Exception as record_exc:  # pragma: no cover - preserve rollout
                        if recorded_tool_call is not None:
                            raise
                        logger.warning(f"failed to record tool error for verifier replay: {record_exc}")
            return (
                ToolResponse(
                    text=error_text,
                ),
                0.0,
                {"tool": tool_name, "error": True, "dispatch_error": True},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    def _initialize_interactions(self, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        return interaction_map
