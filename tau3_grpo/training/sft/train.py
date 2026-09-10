"""Train a full-parameter or LoRA warm-start on 45 complete dialogues."""

from __future__ import annotations

import argparse
import json
import math
import os
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from tau3_grpo.models.compat import (
    load_policy_model,
    lora_target_modules,
    model_family,
    require_training_runtime,
)
from tau3_grpo.paths import PROJECT_ROOT, resolve_project_path
from tau3_grpo.training.sft.dataset import TrajectorySFTDataset, collate_fn_padding


def _resolve(path: str) -> Path:
    return resolve_project_path(path)


def _require_runtime(torch: Any, family: str = "legacy") -> None:
    require_training_runtime(torch, family)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise RuntimeError(
            "the frozen 45-dialogue/effective-batch-8 SFT budget uses one GPU; "
            f"WORLD_SIZE={world_size} would change optimizer-step semantics"
        )


def _load_tools(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [entry["tool_schema"] for entry in payload["tools"]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Airline assistant-only SFT")
    parser.add_argument("--method", choices=("full", "lora"), default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/train/sft/qwen25_lora.yaml",
    )
    parser.add_argument(
        "--model-name-or-path",
        default=None,
        help="runtime override for an offline local copy of the frozen base model",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=None,
        help="memory-safe micro-batch override; effective batch must remain eight",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="paired accumulation override; effective batch must remain eight",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--report-to", choices=("swanlab", "none"),
                        default=os.environ.get("TAU3_TRACKING_BACKEND", "swanlab"))
    parser.add_argument("--run-name", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    import torch
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

    model_name = args.model_name_or_path or config["model"]["name_or_path"]
    family = model_family(model_name)
    _require_runtime(torch, family)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tools = _load_tools(_resolve(config["data"]["tool_config"]))
    max_length = args.max_length or int(config["data"]["max_length"])
    train_dataset = TrajectorySFTDataset(
        _resolve(config["data"]["train_jsonl"]),
        tokenizer,
        tools=tools,
        max_length=max_length,
        expected_size=45,
    )
    validation_dataset = TrajectorySFTDataset(
        _resolve(config["data"]["validation_jsonl"]),
        tokenizer,
        tools=tools,
        max_length=max_length,
        expected_size=5,
    )

    train_config = config["train"]
    method = args.method or train_config.get("method", "lora")
    if method not in {"full", "lora"}:
        raise ValueError(f"unsupported SFT method: {method}")
    per_device_batch = (
        args.per_device_batch_size
        if args.per_device_batch_size is not None
        else int(train_config["per_device_batch_size"])
    )
    accumulation = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else int(train_config["gradient_accumulation_steps"])
    )
    epochs = int(train_config["num_epochs"])
    effective_batch = per_device_batch * accumulation
    if effective_batch != 8:
        raise ValueError(f"frozen SFT effective batch is 8, got {effective_batch}")
    expected_steps = math.ceil(math.ceil(45 / per_device_batch) / accumulation) * epochs
    if expected_steps != 30:
        raise ValueError(f"frozen SFT schedule must produce about 30 steps, got {expected_steps}")

    set_seed(int(train_config["seed"]))
    model = load_policy_model(
        model_name,
        # Full SFT keeps FP32 trainable weights and Adam states; Trainer's BF16
        # autocast controls forward precision. LoRA keeps the BF16 frozen base.
        torch_dtype=torch.float32 if method == "full" else torch.bfloat16,
        attn_implementation=config["model"].get("attn_implementation", "flash_attention_2"),
        trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
    )
    model.config.use_cache = False
    if method == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        lora = config["lora"]
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(lora["r"]),
                lora_alpha=int(lora["alpha"]),
                lora_dropout=float(lora["dropout"]),
                bias="none",
                target_modules=lora_target_modules(model, lora["target_modules"]),
            ),
        )
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
    print(f"SFT method={method}; trainable={trainable_parameters:,}; total={total_parameters:,}")

    output_dir = _resolve(str(args.output_dir or config["output"]["dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=accumulation,
        learning_rate=float(train_config["learning_rate"]),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(train_config.get("warmup_ratio", 0.05)),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
        max_grad_norm=float(train_config.get("max_grad_norm", 1.0)),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=int(train_config.get("logging_steps", 1)),
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        load_best_model_at_end=bool(train_config.get("load_best_model_at_end", False)),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=int(train_config["seed"]),
        data_seed=int(train_config["seed"]),
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer_class = Trainer
    if family == "qwen35":
        from tau3_grpo.training.sft.trainer import Qwen35SFTTrainer

        trainer_class = Qwen35SFTTrainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=partial(collate_fn_padding, pad_token_id=tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    tracking_run = None
    if args.report_to == "swanlab":
        from tau3_grpo.tracking.swanlab import (
            experiment_name,
            log_sft_dataset,
            sft_callback,
            start_run,
        )

        tracking_run = start_run(
            name=args.run_name or experiment_name(stage="sft", model=model_name, method=method,
                lr=float(train_config["learning_rate"]), seed=int(train_config["seed"]),
                rank=int(config.get("lora", {}).get("r", 16)), epochs=epochs),
            config={"stage": "sft", "method": method, "model": model_name,
                    "seed": int(train_config["seed"]),
                    "training": config, "trainable_parameters": trainable_parameters,
                    "effective_batch_size": effective_batch},
            output=output_dir,
        )
        trainer.add_callback(sft_callback(tracking_run))
        for split, source in (("train", "train_jsonl"), ("validation", "validation_jsonl")):
            log_sft_dataset(tracking_run, _resolve(config["data"][source]),
                            output_dir / "swanlab-artifacts", split)
    baseline_metrics = (
        trainer.evaluate() if train_config.get("evaluate_before_train", False) else None
    )
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    validation_metrics = trainer.evaluate()
    if training_args.load_best_model_at_end and not math.isclose(
        validation_metrics["eval_loss"], trainer.state.best_metric, rel_tol=1e-4, abs_tol=1e-5
    ):
        raise RuntimeError("restored checkpoint validation loss does not match the best checkpoint")
    trainer.save_model(str(output_dir))
    trainer.save_state()
    tokenizer.save_pretrained(str(output_dir))
    summary = {
        "model_name_or_path": model_name,
        "model_family": family,
        "sft_method": method,
        "loss_projection": "supervised_positions" if family == "qwen35" else "native",
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "enable_thinking": False if family == "qwen35" else None,
        "expected_optimizer_steps": expected_steps,
        "actual_optimizer_steps": trainer.state.global_step,
        "per_device_batch_size": per_device_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": effective_batch,
        "train": train_dataset.token_stats(),
        "validation": validation_dataset.token_stats(),
        "train_loss": float(result.training_loss),
        "learning_rate": float(train_config["learning_rate"]),
        "seed": int(train_config["seed"]),
        "baseline_validation": baseline_metrics,
        "validation_metrics": validation_metrics,
        "best_validation_loss": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "metrics": result.metrics,
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if tracking_run is not None:
        from tau3_grpo.tracking.swanlab import scalar_metrics

        tracking_run.log(scalar_metrics(summary, "summary"), step=trainer.state.global_step)
        for filename in ("train_summary.json", "trainer_state.json"):
            tracking_run.save(str(output_dir / filename), base_path=str(output_dir), policy="now")
        tracking_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
