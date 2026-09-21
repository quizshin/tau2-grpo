"""Config-backed compatibility defaults; no framework imports or GPU operations.

Keep shell's historical ``${VAR:-default}`` semantics (empty counts as unset).
The public launcher's inherited-environment precedence is unchanged. This module
only fills the final compatibility layer, before native Hydra CLI overrides.
"""
from __future__ import annotations

import argparse
import os
import shlex

from tau3_grpo.configuration import load_config, resolve_arm
from tau3_grpo.paths import CODE_ROOT

BASE = CODE_ROOT / "configs/train/rl/base.yaml"
QWEN35 = CODE_ROOT / "configs/runtime/rl_qwen35.yaml"
ARMS = CODE_ROOT / "configs/experiments/arms.yaml"
BASE_FIELDS = {
    "DATA_SPLIT_SEED": ("data", "split_seed"),
    "GROUP_SIZE": ("rollout", "group_size"),
    "GROUPS_PER_UPDATE": ("rollout", "groups_per_update"),
    "LR": ("optim", "lr"),
    "KL_COEF": ("optim", "kl_coef"),
    "TRAIN_TEMP": ("rollout", "temperature_train"),
    "EVAL_TEMP": ("rollout", "temperature_eval"),
    "MAX_USER_TURNS": ("rollout", "max_user_turns"),
    "MAX_ASSISTANT_TURNS": ("rollout", "max_assistant_turns"),
    "POLICY_GPUS": ("resources", "policy_gpus"),
    "TOTAL_UPDATES": ("schedule", "screening_updates"),
    "ROLLOUT_TP": ("rollout", "tensor_model_parallel_size"),
    "PPO_MINI_GROUPS": ("optim", "ppo_mini_groups"),
}


def defaults(profile: str, inherited: dict[str, str], *, size: str = "0.8B") -> dict[str, str]:
    if profile == "base":
        config = load_config(BASE)
        values = {key: config[section][field] for key, (section, field) in BASE_FIELDS.items()}
        values.update(config["runtime"]["environment"])
    elif profile == "qwen35":
        config = load_config(QWEN35)
        values = dict(config["launch"]["environment"])
        buckets = config["transfer_bucket_megabytes"]
        values["UPDATE_WEIGHTS_BUCKET_MEGABYTES"] = buckets.get(size, buckets["default"])
    else:
        raise ValueError(f"unknown compatibility profile: {profile}")
    return {key: inherited.get(key) or str(value).replace("e-0", "e-")
            for key, value in values.items()}


def arm_environment(arm: str, inherited: dict[str, str]) -> dict[str, str]:
    config = resolve_arm(arm, load_config(ARMS))
    values = {"ESTIMATOR": config["adv_estimator"],
              "DF_ENABLE": str(config["dynamic_filter"]["enable"]).lower(),
              "ANCHOR_MODE": config["anchors"]["mode"]}
    return {("ADV_ESTIMATOR" if key == "ESTIMATOR" else key):
            inherited.get(f"TAU3_GRPO_CONFIG_{key}") or value for key, value in values.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("base", "qwen35", "arm", "exec-qwen35"))
    parser.add_argument("--size", default="0.8B")
    parser.add_argument("--arm", default="e0")
    args, extra = parser.parse_known_args(argv)
    if args.profile == "exec-qwen35":
        if extra[:1] == ["--"]:
            extra = extra[1:]
        if len(extra) < 2:
            parser.error("exec-qwen35 requires -- ARM SEED [Hydra overrides]")
        config = load_config(QWEN35)
        command = ["bash", str(CODE_ROOT / "scripts/train/rl/run_base.sh"), *extra[:2],
                   *config["wrapper_overrides"], *extra[2:]]
        os.execvp(command[0], command)
    if extra:
        parser.error(f"unexpected arguments: {extra}")
    values = (arm_environment(args.arm, dict(os.environ)) if args.profile == "arm"
              else defaults(args.profile, dict(os.environ), size=args.size))
    # Quote values as data: inherited paths or config text must never execute shell code.
    for key, value in values.items():
        print(f"export {key}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
