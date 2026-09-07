"""Train the v2-2 Qwen2.5-7B LoRA warm-start on 45 complete dialogues."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from tau3_grpo.paths import PROJECT_ROOT
from tau3_grpo.sft.dataset import TrajectorySFTDataset, collate_fn_padding


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _require_runtime(torch: Any) -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"SFT requires Python 3.12, got {sys.version.split()[0]}")
    if not torch.__version__.startswith("2.8"):
        raise RuntimeError(f"SFT requires torch 2.8, got {torch.__version__}")
    if torch.version.cuda != "12.8" or not torch.cuda.is_available():
        raise RuntimeError(
            f"SFT requires a live CUDA 12.8 GPU, got cuda={torch.version.cuda!r}"
        )
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
    parser = argparse.ArgumentParser(description="Run Airline assistant-only LoRA SFT")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/sft_airline_lora.yaml",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    _require_runtime(torch)
    model_name = args.model_name_or_path or config["model"]["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tools = _load_tools(_resolve(config["data"]["tool_config"]))
    max_length = int(config["data"]["max_length"])
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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=config["model"].get("attn_implementation", "flash_attention_2"),
        trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
    )
    model.config.use_cache = False
    lora = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            bias="none",
            target_modules=list(lora["target_modules"]),
        ),
    )
    model.print_trainable_parameters()

    output_dir = _resolve(config["output"]["dir"])
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
        report_to="none",
        seed=int(train_config["seed"]),
        data_seed=int(train_config["seed"]),
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=partial(collate_fn_padding, pad_token_id=tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    summary = {
        "expected_optimizer_steps": expected_steps,
        "per_device_batch_size": per_device_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": effective_batch,
        "train": train_dataset.token_stats(),
        "validation": validation_dataset.token_stats(),
        "train_loss": float(result.training_loss),
        "metrics": result.metrics,
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
