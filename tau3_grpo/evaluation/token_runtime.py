"""Independent selection via the actual training loop and a token-only endpoint.

No optimizer, Ray cluster, model weights, or GPU initialization occurs here.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from tau3_grpo.models.token_budget import VERSION, default_budget


def prepare_runtime():
    """Load registry targets through Python's import lock before native lookup.

    veRL's dynamic registries publish modules in sys.modules before executing
    their bodies. Preloading prevents concurrent loops seeing a partial class.
    """
    from importlib import import_module

    for module, name in (("tau3_grpo.envs.tools", "Tau3AirlineTool"),
                         ("tau3_grpo.envs.interaction", "Tau3AirlineInteraction")):
        getattr(import_module(module), name)


def load_tokenizer(path):
    from transformers import AutoTokenizer

    directory = Path(path)
    required = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    if not all((directory / name).is_file() for name in required):
        raise ValueError("Token evaluation requires the checkpoint's local tokenizer files")
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    return tokenizer


def tokenizer_identity(path):
    import hashlib

    directory = Path(path)
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")}


def check_context(endpoint, expected):
    """Read-only service capability check; never probes by generating a sample."""
    import urllib.request

    req = urllib.request.Request(endpoint.base_url.rstrip("/") + "/models",
                                 headers={"Authorization": "Bearer " + endpoint.api_key})
    with urllib.request.urlopen(req, timeout=30) as response:
        models = json.load(response).get("data", [])
    matches = [item for item in models if item.get("id") == endpoint.model]
    if len(matches) != 1 or matches[0].get("max_model_len") != expected:
        raise ValueError(f"Service {endpoint.model} must attest max_model_len={expected}")


async def make_loop(*, tokenizer, manager, user, directory, estimator="grpo", evaluation=True):
    """Construct the native loop normally, using the real full-schema registry."""
    prepare_runtime()
    import yaml
    from omegaconf import OmegaConf
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

    from tau3_grpo.envs.generate_tool_config import FULL_SCHEMA, write_config
    from tau3_grpo.envs.interaction import Tau3AirlineInteraction
    from tau3_grpo.envs.registry import SESSIONS
    from tau3_grpo.envs.session import TrajectorySession
    from tau3_grpo.evaluation.verifier import verify_trajectory
    from tau3_grpo.integrations.verl.token_budget import configure

    budget = default_budget()
    tools = write_config(Path(directory) / "tools.yaml", FULL_SCHEMA)
    user_config = {"user_model": user.model, "user_base_url": user.base_url,
                   "user_temperature": user.temperature, "user_max_tokens": budget.user_per_turn,
                   "user_llm_args": {"api_key": user.api_key}, "strict_replay": True}
    interactions = Path(directory) / "interaction.yaml"
    # Never persist credentials to the registry file. The instance receives them below.
    interactions.write_text(yaml.safe_dump({"interaction": [{"name": "tau3_airline",
        "class_name": "tau3_grpo.envs.interaction.Tau3AirlineInteraction", "config": {}}]}))
    config = OmegaConf.create({"tau3_token_protocol": VERSION, "tau3_evaluation_only": evaluation,
        "algorithm": {"adv_estimator": estimator}, "actor_rollout_ref": {"model": {}, "rollout": {
            "prompt_length": budget.prompt, "response_length": budget.response,
            "max_model_len": budget.context, "multi_turn": {
                "max_user_turns": 15, "max_assistant_turns": 15, "max_parallel_calls": 1,
                "tool_execution_mode": "sequential", "max_tool_response_length": 65536,
                "tool_response_truncate_side": "middle", "tool_config_path": str(tools),
                "interaction_config_path": str(interactions), "format": "qwen3_coder"}}}})
    loop = ToolAgentLoop(trainer_config=DictConfigWrap(config), server_manager=manager,
                        tokenizer=tokenizer, processor=None, dataset_cls=None,
                        data_config=DictConfigWrap(OmegaConf.create({
                            "apply_chat_template_kwargs": {"enable_thinking": False}})))

    class EvaluationInteraction(Tau3AirlineInteraction):
        simulation = None

        def _create_session(self, task_id, kwargs, *, session_id):
            # Eval seed is per trial; an inherited training seed cannot override it.
            return TrajectorySession(self.adapted, user_config=self._user_config,
                                     limits=self._limits, session_id=session_id, seed=kwargs["seed"])

        def score_trajectory(self, instance_id, *, termination_reason=None, **kwargs):
            entry = SESSIONS.require(instance_id)
            result = verify_trajectory(entry.session, termination_reason=termination_reason,
                                       strict_replay=True, tool_error_count=entry.tool_error_count,
                                       duration=time.monotonic() - self.started)
            self.simulation = result.simulation
            self.simulation.info = {"initial_db_hash": result.initial_db_hash,
                                    "final_db_hash": result.db_hash, "officially_scored": result.scored}
            return result.to_dict()

    handler = EvaluationInteraction(user_config) if evaluation else Tau3AirlineInteraction(user_config)
    loop.interaction_map = {"tau3_airline": handler}
    loop.token_budget = configure(loop)
    loop.record_turn_facts = True
    return loop, handler


def run_one(*, entry, policy, user, seed, trial, tokenizer, manager=None, request_timeout=120):
    from tau3_grpo.data.parquet_builder import build_row
    from tau3_grpo.data.schema import ArealTaskRecord
    from tau3_grpo.envs.adapter import adapt_record, build_environment, load_flight_db
    from tau3_grpo.evaluation.harness import TOKENS_V4, protocol_metadata, validate_opening
    from tau3_grpo.models.token_endpoint import TokenEndpoint

    adapted = adapt_record(ArealTaskRecord.model_validate(entry.task))
    environment = build_environment(load_flight_db(adapted.db_path))
    row = build_row(entry, policy=environment.get_policy(), split="selection", seed=seed)
    validate_opening(adapted.task, row["extra_info"]["interaction_kwargs"]["initial_user_message"])
    endpoint = manager or TokenEndpoint(policy, timeout=request_timeout)

    async def execute():
        with tempfile.TemporaryDirectory(prefix="tau3-token-eval-") as directory:
            loop, handler = await make_loop(tokenizer=tokenizer, manager=endpoint, user=user,
                                            directory=directory)
            handler.adapted = adapted
            handler.started = time.monotonic()
            output = await loop.run({"temperature": policy.temperature, "top_p": 1.0, "top_k": -1,
                                     "repetition_penalty": 1.0, "logprobs": True, "seed": seed},
                raw_prompt=row["prompt"], extra_info=row["extra_info"],
                tau3_sampling_identity={"trial": trial, "seed": seed, "task_id": entry.task_id})
            simulation = handler.simulation
            if simulation is None:
                raise RuntimeError("Shared token loop did not produce an official simulation")
            simulation.trial = trial
            simulation.info = {**(simulation.info or {}),
                "harness_protocol": protocol_metadata(TOKENS_V4),
                "token_protocol": json.loads(output.extra_fields["token_protocol_json"]),
                "trajectory_facts": json.loads(output.extra_fields["trajectory_facts_json"]),
                "token_budget": asdict(loop.token_budget)}
            return simulation

    return asyncio.run(execute())
