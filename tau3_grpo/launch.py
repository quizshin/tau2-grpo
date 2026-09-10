"""Configuration entrypoint shared by SFT, RL and simulator launchers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from string import Template

import yaml

from tau3_grpo.paths import CODE_ROOT
from tau3_grpo.tracking.swanlab import load_tracking_env, redact_config


def merge(base: dict, update: dict) -> dict:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        elif key == "overrides" and isinstance(value, list):
            result[key] = result.get(key, []) + value
        else:
            result[key] = value
    return result


def load_config(path: Path, stack: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in stack:
        raise ValueError(f"cyclic config include: {path}")
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    result: dict = {}
    for parent in payload.pop("includes", []):
        result = merge(result, load_config(path.parent / parent, (*stack, path)))
    return merge(result, payload)


def prepare(stage: str, path: Path, experiment: str, seed: int | None,
            extra: list[str], inherited: dict[str, str]) -> tuple[list[str], dict, dict]:
    config = load_config(path)
    launch = config.get("launch", {})
    if launch.get("stage") != stage:
        raise ValueError(f"expected launch.stage={stage!r} in {path}")
    env = dict(inherited)
    env["CODE_ROOT"] = str(CODE_ROOT)
    for name, default in (("TAU3_DATA_ROOT", "data"), ("TAU3_MODEL_ROOT", "models"),
                          ("TAU3_RUN_ROOT", "results"), ("TAU3_CACHE_ROOT", ".cache")):
        value = Path(env.get(name, str(CODE_ROOT / default))).expanduser()
        env[name] = str(value if value.is_absolute() else CODE_ROOT / value)
    for key, value in launch.get("environment", {}).items():
        # Explicit shell/.env settings override profile defaults.
        if key not in env:
            env[key] = Template(str(value)).substitute(env)
    script = (CODE_ROOT / launch["entrypoint"]).resolve()
    if CODE_ROOT / "scripts" not in script.parents or not script.is_file():
        raise ValueError(f"invalid launch entrypoint: {script}")
    env["PYTHONPATH"] = os.pathsep.join((str(CODE_ROOT), str(CODE_ROOT / "verl"),
                                        str(CODE_ROOT / "tau2-bench/src"),
                                        env.get("PYTHONPATH", "")))
    env["TAU3_TRACKING_ENV_LOADED"] = "1"
    command = ["bash", str(script)]
    algorithm_overrides = []
    if stage == "rl":
        arms = load_config(CODE_ROOT / "configs/experiments/arms.yaml")
        if experiment in arms["arms"]:
            arm = arms["arms"][experiment]
        else:
            ablation = arms["ablations"][experiment]
            arm = merge(arms["arms"][ablation["base_arm"]], ablation)
        env["TAU3_GRPO_CONFIG_ESTIMATOR"] = arm["adv_estimator"]
        env["TAU3_GRPO_CONFIG_DF_ENABLE"] = str(arm["dynamic_filter"]["enable"]).lower()
        env["TAU3_GRPO_CONFIG_ANCHOR_MODE"] = arm["anchors"]["mode"]
        for key, value in arm.get("gigpo", {}).items():
            algorithm_overrides.append(f"++algorithm.gigpo.{key}={value}")
        snapshot_arm = arm
        command += [experiment, str(seed if seed is not None else launch.get("seed", 42))]
    elif stage == "sft":
        # Native SFT schema stays in this same YAML; no duplicated training settings.
        command += ["--config", str(path.resolve())]
        env.setdefault("SFT_MODEL_NAME_OR_PATH", env.get("MODEL_PATH", config["model"]["name_or_path"]))
    command += [Template(str(x)).substitute(env) for x in launch.get("overrides", [])]
    command += algorithm_overrides + extra
    snapshot = {"config_file": str(path.resolve()), "configuration": config,
                "command": command, "environment": {k: env[k] for k in
                    {*launch.get("environment", {}), "TAU3_DATA_ROOT", "TAU3_MODEL_ROOT",
                     "TAU3_RUN_ROOT", "TAU3_CACHE_ROOT"}}, "stage": stage}
    if stage == "rl":
        snapshot["experiment"] = snapshot_arm
    return command, env, redact_config(snapshot)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sft", "rl", "simulator"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment", default="e0",
                        choices=("e0", "e1", "e2", "e3", "e2_db_hash_only", "e2_similarity"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true", help="resolve config without starting services or training")
    args, extra = parser.parse_known_args(argv)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    load_tracking_env()
    path = args.config if args.config.is_absolute() else CODE_ROOT / args.config
    command, env, snapshot = prepare(args.stage, path, args.experiment, args.seed, extra, dict(os.environ))
    if args.dry_run:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
        return 0
    return subprocess.call(command, env=env, cwd=CODE_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
