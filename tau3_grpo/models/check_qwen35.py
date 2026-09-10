"""Read-only model/tokenizer audit; GPU mode also checks the candidate runtime.

This command does not start a Ray cluster, load policy weights, or train.
"""

from __future__ import annotations

import argparse
import importlib
import json
from importlib.metadata import version
from pathlib import Path

import yaml

from tau3_grpo.models.compat import model_family, policy_model_class, require_training_runtime
from tau3_grpo.paths import CONFIG_ROOT, SFT_DATA_ROOT
from tau3_grpo.training.sft.dataset import TrajectorySFTDataset
from tau3_grpo.utils.hashing import sha256_file


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--audit-sft", action="store_true")
    parser.add_argument(
        "--train", type=Path, default=SFT_DATA_ROOT / "airline_sft_train_seed42.jsonl"
    )
    parser.add_argument(
        "--validation", type=Path, default=SFT_DATA_ROOT / "airline_sft_validation_seed42.jsonl"
    )
    parser.add_argument("--tool-config", type=Path, default=CONFIG_ROOT / "envs/tool_config.yaml")
    parser.add_argument("--max-length", type=int, default=24576)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoConfig, AutoTokenizer

    if model_family(args.model) != "qwen35":
        raise ValueError("Expected a dense Qwen3.5 model/config path")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
    if config.model_type != "qwen3_5":
        raise ValueError(f"Expected qwen3_5, got {config.model_type}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    # Exercise the actual template before a GPU or data audit.
    from tau3_grpo.training.sft.dataset import build_supervised_example

    example = build_supervised_example(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": "You are welcome."},
        ],
        tokenizer,
    )
    report = {
        "model": args.model,
        "model_type": config.model_type,
        "auto_model_class": policy_model_class(config).__name__,
        "transformers": version("transformers"),
        "torch": torch.__version__,
        "template_label_tokens": example["n_label_tokens"],
        "gpu_runtime_checked": False,
        "gpu_training_verified": False,
    }
    if args.gpu:
        model_dir = Path(args.model)
        if model_dir.is_dir() and not any(model_dir.glob("*.safetensors")):
            raise FileNotFoundError("GPU validation needs a complete local checkpoint, not tokenizer-only assets")
        require_training_runtime(torch, "qwen35")
        if version("vllm") != "0.20.0":
            raise RuntimeError("Qwen3.5 rollout profile requires vllm==0.20.0")
        # Catch dependency/API import errors before the launcher allocates actors.
        for name in (
            "verl.workers.fsdp_workers",
            "verl.workers.rollout.vllm_rollout.vllm_async_server",
        ):
            importlib.import_module(name)
        report.update(
            gpu_runtime_checked=True,
            cuda=torch.version.cuda,
            gpu_count=torch.cuda.device_count(),
            vllm=version("vllm"),
        )
    if args.audit_sft:
        payload = yaml.safe_load(args.tool_config.read_text(encoding="utf-8"))
        tools = [entry["tool_schema"] for entry in payload["tools"]]
        report["sft"] = {}
        for split, path, count in (("train", args.train, 45), ("validation", args.validation, 5)):
            dataset = TrajectorySFTDataset(
                path, tokenizer, tools=tools, max_length=args.max_length, expected_size=count
            )
            report["sft"][split] = {
                "sha256": sha256_file(path),
                **dataset.token_stats(),
                "max_tokens": max(row["n_total_tokens"] for row in dataset.examples),
            }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
