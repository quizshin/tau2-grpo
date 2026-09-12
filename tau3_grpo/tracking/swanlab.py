"""SwanLab metrics, readable trajectories and explicit source artifacts.

The SDK is imported only when tracking is enabled. No credentials are stored in
configs or artifacts; authentication uses SwanLab's login store/environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import zipfile
from decimal import Decimal
from pathlib import Path
from typing import Any

from tau3_grpo.paths import PROJECT_ROOT


def load_tracking_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(os.environ.get("TAU3_ENV_FILE", PROJECT_ROOT / ".env"), override=False)


def experiment_name(*, stage: str, model: str, method: str, lr: float, seed: int,
                    rank: int = 16, epochs: int = 5, arm: str = "E0",
                    groups: int = 16, group_size: int = 8, updates: int = 40) -> str:
    """Match code_pytrio's stage/model/method/LR/budget/seed naming convention."""
    short = model.rstrip("/").rsplit("/", 1)[-1]
    train_mode = f"lora{rank}" if method == "lora" else "full"
    common = f"{short}-{train_mode}-lr{format(Decimal(str(lr)).normalize(), 'E').lower()}"
    if stage == "sft":
        return f"sft-airline-{common}-ep{epochs}-s{seed}"
    arm = arm.upper()
    label = {"E0": "base", "E1": "DF", "E2": "GiGPO", "E3": "DF-GiGPO"}.get(arm, arm)
    return f"rl-{arm}-{label}-{common}-g{groups}x{group_size}-u{updates}-s{seed}"


def rl_experiment_name(config: Any) -> str:
    """Name the resolved RL configuration, including caller Hydra overrides."""
    if config.trainer.experiment_name != "tau3-auto":
        return config.trainer.experiment_name
    actor = config.actor_rollout_ref
    rank = int(actor.model.get("lora_rank", 0))
    return experiment_name(
        stage="rl", model=os.environ.get("TAU3_POLICY_DISPLAY_MODEL", actor.model.path),
        method="lora" if rank > 0 else "full", rank=rank,
        lr=float(actor.actor.optim.lr), seed=int(config.data.seed),
        arm=os.environ.get("TAU3_GRPO_ARM", "e0"),
        groups=int(config.data.train_batch_size), group_size=int(actor.rollout.n),
        updates=int(config.trainer.total_training_steps),
    )


def setup_rl_charts(loggers: Any) -> None:
    if "swanlab" not in loggers:
        return
    import swanlab

    for prefix in ("actor", "critic", "train", "training", "perf", "timing_s", "val",
                   "val-core", "val-aux", "rollout", "dynamic_filter", "gigpo", "anchors", "budget"):
        swanlab.define_metric(prefix + "/*", x_axis="trainer/global_step", section_name=prefix)
    swanlab.define_metric("cases/*", section_name="cases")
    swanlab.define_metric("trainer/global_step", hidden=True)


def rl_tracking_metadata(config: dict) -> tuple[dict, dict]:
    """Match the existing pytrio dashboard's trainer.phase=rl filter."""
    config = redact_config(config)
    trainer = config.setdefault("trainer", {})
    trainer.update(phase="rl", backend="veRL")
    config["stage"] = "rl"
    actor = config.get("actor_rollout_ref", {})
    model = actor.get("model", {})
    rank = int(model.get("lora_rank", 0))
    method = "lora" if rank or model.get("lora_adapter_path") else "full"
    arm = os.environ.get("TAU3_GRPO_ARM", "e0").upper()
    name = os.environ.get("TAU3_POLICY_DISPLAY_MODEL", model.get("path", "unknown"))
    config["actor"] = {"model": name, "training_mode": method, "lora_rank": rank,
                       "optim": actor.get("actor", {}).get("optim", {})}
    trainer["arm"] = arm
    return config, {"job_type": "rl", "group": f"RL-{arm}",
                    "workspace": os.environ.get("SWANLAB_WORKSPACE") or None,
                    "tags": ["RL", arm, "veRL", method, name.rstrip('/').rsplit('/', 1)[-1]]}


def redact_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): "[REDACTED]" if str(k).lower() in {
                "api_key", "password", "secret", "access_token", "auth_token", "token"
            } or str(k).lower().endswith("_api_key") else redact_config(v)
            for k, v in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [redact_config(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def scalar_metrics(values: dict[str, Any], prefix: str = "") -> dict[str, float | int]:
    result = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            result.update(scalar_metrics(value, name))
        elif isinstance(value, (int, float)) and math.isfinite(value):
            result[name] = value
    return result


def sft_metrics(logs: dict[str, Any]) -> dict[str, float | int]:
    """Use identical chart names for live callbacks and historical replay."""
    result = {}
    for key, value in scalar_metrics(logs).items():
        if key == "step":
            continue
        if key.startswith("eval_"):
            result[f"val/{key[5:]}"] = value
        else:
            result[f"train/{'lr' if key == 'learning_rate' else key}"] = value
    return result


def create_source_archive(output: Path, code_root: Path | None = None) -> Path:
    """Archive source directories only; never scan environments/data/logs/credentials."""
    root = code_root or PROJECT_ROOT
    selected = []
    for directory in (
        "tau3_grpo", "configs",
        "scripts", "verl/verl", "tau2-bench/src",
    ):
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root)
            if path.is_file() and not path.is_symlink() and not any(
                part.startswith(".") or part == "__pycache__" for part in relative.parts
            ) and path.suffix in {".py", ".sh", ".yaml", ".yml", ".toml", ".json"}:
                selected.append(path)
    for filename in (
        "THIRD_PARTY_SOURCES.md", "env_info/versions.lock", "env_info/qwen35-constraints.txt",
        "pyproject.toml", "verl/setup.py",
    ):
        if (root / filename).is_file():
            selected.append(root / filename)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "source-sft-rl.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(set(selected)):
            handle.write(path, path.relative_to(root))
    return archive


def attach_sources(run: Any, output: Path) -> None:
    archive = create_source_archive(output)
    run.save(str(archive), base_path=str(output), policy="now")
    manifest = {
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "description": "Current source snapshot at logging time, including instrumentation/fixes",
    }
    (output / "source-snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
    run.save(str(output / "source-snapshot.json"), base_path=str(output), policy="now")


def run_url(run: Any) -> str | None:
    return run.url if run.mode == "online" else None


def start_run(*, name: str, config: dict, output: Path, job_type: str = "sft") -> Any:
    load_tracking_env()
    import swanlab

    tracking_config = redact_config(config)
    tracking_config.setdefault("trainer", {}).setdefault("phase", job_type)
    run = swanlab.init(
        project=os.environ.get("TAU3_SWANLAB_PROJECT", "tau3-grpo-pytrio"),
        workspace=os.environ.get("SWANLAB_WORKSPACE") or None,
        name=name, job_type=job_type, config=tracking_config,
        mode=os.environ.get("SWANLAB_MODE", "online"),
        log_dir=os.environ.get("SWANLAB_LOG_DIR", str(output / "swanlog")),
        public=False,
        group=f"{job_type}-seed{config.get('seed', 42)}",
        tags=[job_type.upper(), "airline", "veRL", str(config.get("method", "full"))],
    )
    for prefix in ("train", "val"):
        swanlab.define_metric(prefix + "/*", x_axis="train/global_step", section_name=prefix)
    swanlab.define_metric("train/global_step", hidden=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "swanlab-run.json").write_text(json.dumps(
        {"name": run.name, "url": run_url(run), "local_dir": str(run.dir)}, indent=2
    ) + "\n")
    attach_sources(run, output / "swanlab-artifacts")
    return run


def conversation_text(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return value
    if isinstance(value, dict):
        messages = value.get("messages")
        if messages is None:
            if "prompt" in value and "response" in value:
                return f"[Initial context]\n{value['prompt']}\n\n[Generated interaction]\n{value['response']}"
            return json.dumps(value, ensure_ascii=False, indent=2)
    elif isinstance(value, list):
        messages = value
    else:
        return str(value)
    parts = []
    for index, message in enumerate(messages):
        parts.append(f"[{index}] {message.get('role', 'unknown')}")
        if message.get("content") is not None:
            content = message["content"]
            parts.append(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
        if message.get("tool_calls"):
            parts.append("Tool calls:\n" + json.dumps(message["tool_calls"], ensure_ascii=False, indent=2))
        if message.get("tool_call_id"):
            parts.append("tool_call_id=" + str(message["tool_call_id"]))
    return "\n\n".join(parts)


def sample_indices(rows: list[dict], count: int, seed: int) -> list[int]:
    """Deterministic success/failure sampling; excluded padding never reaches here."""
    rng = random.Random(seed)
    positive = [i for i, row in enumerate(rows) if row.get("reward", 0) > 0]
    other = [i for i, row in enumerate(rows) if row.get("reward", 0) <= 0]
    rng.shuffle(positive)
    rng.shuffle(other)
    selected = []
    seen_tasks = set()

    def task(index):
        return str(rows[index]["task_id"]) if rows[index].get("task_id") else ("row", index)

    def take(pool, size):
        for _ in range(min(size, len(pool))):
            index = next((i for i in pool if task(i) not in seen_tasks), pool[0])
            pool.remove(index)
            selected.append(index)
            seen_tasks.add(task(index))

    take(positive, count // 2)
    take(other, count - count // 2)
    rest = [i for i in range(len(rows)) if i not in selected]
    rng.shuffle(rest)
    take(rest, count - len(selected))
    return selected


def log_examples(run: Any, rows: list[dict], *, key: str, step: int,
                 output: Path, count: int = 4) -> None:
    import swanlab

    if count <= 0 or not rows:
        return
    selected = [rows[i] for i in sample_indices(rows, count, seed=42 + step)]
    output.mkdir(parents=True, exist_ok=True)
    filename = output / f"{key.replace('/', '-')}-step{step:06d}.jsonl"
    filename.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n"
                                for row in selected))
    table = swanlab.echarts.Table()
    table_rows = []
    for row in selected:
        text = conversation_text(row.get("simulation") or row.get("trajectory") or row)
        table_rows.append([step, row.get("task_id", ""), row.get("reward"),
                           row.get("termination_reason", ""), text[:24000], len(text) > 24000])
    table.add(headers=["step", "task_id", "reward", "termination", "conversation", "preview_truncated"],
              rows=table_rows)
    payload = {key: table}
    if key in {"train/rollout_examples", "cases/recovered_dialogues"}:
        # Keep the table and expose the same media keys as the pytrio dashboard.
        prefix = "cases/recovered_" if key == "cases/recovered_dialogues" else "cases/"
        payload[prefix + "batch"] = swanlab.Text(
            "task_id | reward | termination\n" + "\n".join(
                f"{row.get('task_id', '')} | {row.get('reward')} | {row.get('termination_reason', '')}"
                for row in rows
            ), caption=f"RL update {step} · {len(rows)} real trajectories",
        )
        seen = set()
        for row in selected:
            task_id = row.get("task_id", "")
            if task_id in seen:
                continue
            seen.add(task_id)
            text = conversation_text(row.get("simulation") or row.get("trajectory") or row)
            preview = text[:24000] + ("\n[Preview truncated; full text in JSONL artifact]" if len(text) > 24000 else "")
            payload[f"{prefix}preview_{len(seen)}"] = swanlab.Text(
                preview, caption=f"{task_id} · reward {row.get('reward')} · update {step}",
            )
            if len(seen) == 4:
                break
    run.log(payload, step=step)
    run.save(str(filename), base_path=str(output), policy="now")


def sft_callback(run: Any) -> Any:
    from transformers import TrainerCallback

    class Callback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero:
                metrics = sft_metrics(logs or {})
                if metrics:
                    metrics["train/global_step"] = state.global_step
                    run.log(metrics, step=state.global_step)

    return Callback()


def log_sft_dataset(run: Any, path: Path, output: Path, split: str) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    log_examples(run, rows, key=f"dataset/{split}_reference_dialogues", step=0,
                 output=output, count=3)
    run.save(str(path), base_path=str(path.parent), policy="now")


def log_rl_examples(*, batch: Any, tokenizer: Any, update_index: int,
                    loggers: Any, output_dir: str) -> None:
    if "swanlab" not in loggers:
        return
    import swanlab

    run = swanlab.get_run()
    if run is None:
        raise RuntimeError("SwanLab logger was selected but no run is active")
    output = Path(output_dir) / "swanlab-artifacts"
    marker = output / "swanlab-source-run.txt"
    if not marker.exists() or marker.read_text() != str(run.dir):
        attach_sources(run, output)
        marker.write_text(str(run.dir))
    interval = int(os.environ.get("SWANLAB_ROLLOUT_INTERVAL", "1"))
    count = int(os.environ.get("SWANLAB_ROLLOUT_SAMPLES", "4"))
    if interval <= 0 or count <= 0 or update_index % interval:
        return
    non_tensor = batch.non_tensor_batch
    scores = batch.batch["token_level_scores"].sum(-1).detach().cpu().tolist()
    rows = []
    for index, reward in enumerate(scores):
        if non_tensor.get("tau3_is_padding", [False] * len(scores))[index]:
            continue
        row = {"reward": reward}
        for name in ("task_id", "termination_reason", "failure_category", "trajectory_json"):
            if name in non_tensor:
                value = non_tensor[name][index]
                if name == "trajectory_json":
                    row["trajectory"] = json.loads(value) if isinstance(value, str) else value
                else:
                    row[name] = str(value)
        trajectory = row.get("trajectory")
        has_dialogue = (
            isinstance(trajectory, dict)
            and ((isinstance(trajectory.get("messages"), list) and bool(trajectory["messages"]))
                 or ("prompt" in trajectory and "response" in trajectory))
        ) or (isinstance(trajectory, list) and bool(trajectory)
              and all(isinstance(message, dict) and "role" in message for message in trajectory))
        if not has_dialogue:
            if trajectory is not None:
                row["trajectory_metadata"] = row.pop("trajectory")
            prompt_ids = batch.batch["prompts"][index]
            response_ids = batch.batch["responses"][index]
            if "attention_mask" in batch.batch:
                attention_mask = batch.batch["attention_mask"][index].bool()
                prompt_length = len(prompt_ids)
                prompt_ids = prompt_ids[attention_mask[:prompt_length]]
                # response_mask excludes user/tool turns from the loss; keep them here.
                response_ids = response_ids[attention_mask[prompt_length:]]
            row["trajectory"] = {
                "prompt": tokenizer.decode(prompt_ids, skip_special_tokens=False),
                "response": tokenizer.decode(response_ids, skip_special_tokens=False),
            }
        rows.append(row)
    log_examples(run, rows, key="train/rollout_examples", step=update_index,
                 output=output, count=count)
