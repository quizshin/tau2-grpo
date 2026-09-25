"""Driver-side pre-update guard. Fault snapshots are not completed checkpoints."""
from __future__ import annotations

import json
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from tau3_grpo.models.generation_guard import GenerationGuard


@dataclass(frozen=True)
class UpdateGuard:
    version: str = "tau3_update_guard_v1"
    mode: str = "off"
    repetition_fraction: float | None = None
    length_fraction: float | None = None
    consecutive_steps: int = 1

    def __post_init__(self):
        if self.version != "tau3_update_guard_v1" or self.mode not in {"off", "warn", "halt"}:
            raise ValueError("Invalid update guard version/mode")
        if type(self.consecutive_steps) is not int or self.consecutive_steps < 1:
            raise ValueError("consecutive_steps must be positive")
        for value in (self.repetition_fraction, self.length_fraction):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or not 0 < value <= 1):
                raise ValueError("Guard fractions must be in (0,1]")
        if self.mode != "off" and self.repetition_fraction is None and self.length_fraction is None:
            raise ValueError("Enabled update guard requires explicit calibrated thresholds")


class HarnessUpdateHalt(RuntimeError):
    """The candidate batch was persisted and has not entered optimizer update."""


def validate_guard_configuration(config):
    rollout = config["actor_rollout_ref"]["rollout"]
    generation = GenerationGuard.from_config(rollout.get("generation_guard"))
    update = UpdateGuard(**dict(config.get("tau3_update_guard") or {}))
    if generation.mode != "off" or update.mode != "off":
        if config.get("tau3_token_protocol") != "tau3_token_budget_v1":
            raise ValueError("Harness guards require the shared Tau3 token protocol")
        if rollout.get("name") != "vllm" or not (rollout.get("multi_turn") or {}).get("enable"):
            raise ValueError("Harness guards require the vLLM multi-turn loop")
    if update.mode != "off" and update.repetition_fraction is not None and generation.mode == "off":
        raise ValueError("Repetition update guard requires observe or abort generation telemetry")
    if generation.mode != "off" and not rollout.get("calculate_log_probs", False):
        raise ValueError("Guarded generation requires sampled-token logprobs")


def batch_health(non_tensor_batch):
    rows = non_tensor_batch.get("trajectory_facts_json")
    if rows is None:
        raise ValueError("Update guard requires complete trajectory facts")
    padding = non_tensor_batch.get("tau3_is_padding", [False] * len(rows))
    if len(rows) != len(padding):
        raise ValueError("Guard metadata/padding length mismatch")
    count = repeated = length = successes = 0
    identities = set()
    for raw, pad in zip(rows, padding, strict=True):
        if pad:
            continue
        facts = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(facts, dict) or facts.get("schema") != "tau3_trajectory_facts_v1":
            raise ValueError("Invalid trajectory facts for update guard")
        identity = facts["identity"]["trajectory_id"]
        if not identity or identity in identities:
            raise ValueError("Missing/duplicate candidate trajectory identity")
        identities.add(identity)
        count += 1
        repeated += any(bool(t.get("generation_guard")) for t in facts["turns"])
        length += any(t.get("finish_reason") == "length" for t in facts["turns"])
        reward = facts["terminal"].get("reward")
        if reward is None or not math.isfinite(float(reward)):
            raise ValueError("Guard requires a finite recorded outcome")
        successes += abs(float(reward) - 1) <= 1e-6
    if not count:
        raise ValueError("Update guard received no real candidate trajectories")
    return {"candidates": count, "repeated": repeated, "length": length, "successes": successes,
            "repetition_fraction": repeated / count, "length_fraction": length / count}


def guard_decision(health, config, *, previous=None, step):
    previous = previous or {}
    reasons = [key for key in ("repetition_fraction", "length_fraction")
               if getattr(config, key) is not None and health[key] >= getattr(config, key)]
    streak = ((previous.get("streak", 0) if previous.get("step") == step - 1 else 0) + 1) if reasons else 0
    return {"step": step, "streak": streak, "reasons": reasons, "health": health,
            "halt": config.mode == "halt" and streak >= config.consecutive_steps}


def enforce_update_guard(batch, config, *, step, output_dir, previous=None, resolved_config=None):
    settings = UpdateGuard(**dict(config or {}))
    if settings.mode == "off":
        return previous, {}
    decision = guard_decision(batch_health(batch.non_tensor_batch), settings, previous=previous, step=step)
    metrics = {f"harness_guard/{k}": v for k, v in decision["health"].items()}
    metrics.update({"harness_guard/warning": int(bool(decision["reasons"])),
                    "harness_guard/streak": decision["streak"]})
    if decision["halt"]:
        root = Path(output_dir) / "harness-faults"
        root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=f"before-step-{step:06d}-", dir=root))
        # Preserve the complete real batch (including masks/log-probs/advantages),
        # never reuse it automatically and never label it a recovery checkpoint.
        batch.save_to_disk(str(directory / "pending-batch.pkl"))
        from omegaconf import OmegaConf

        from tau3_grpo.tracking.swanlab import redact_config

        if OmegaConf.is_config(resolved_config):
            resolved_config = OmegaConf.to_container(resolved_config, resolve=True)
        receipt = {"schema": "tau3_harness_fault_v1", "status": "halted_before_update",
                   "pending_step": step, "last_completed_outer_step": step - 1,
                   "optimizer_update_started_for_pending_batch": False,
                   "is_resumable_checkpoint": False, "guard": asdict(settings),
                   "decision": decision, "resolved_config": redact_config(resolved_config)}
        (directory / "fault.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        raise HarnessUpdateHalt(f"Harness guard stopped before update {step}; evidence: {directory}")
    return decision, metrics
