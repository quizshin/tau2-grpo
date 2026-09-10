"""Export full SFT or merge LoRA SFT into a BF16 Hugging Face checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tau3_grpo.models.compat import load_policy_model, model_family


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the Airline SFT checkpoint")
    parser.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", type=Path, required=True,
                        help="LoRA adapter directory or a full SFT checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("checkpoint merge requires Python 3.12")
    import torch
    from transformers import AutoTokenizer

    if model_family(args.base) == "legacy" and not torch.__version__.startswith("2.8"):
        raise RuntimeError(f"checkpoint merge requires torch 2.8, got {torch.__version__}")
    if args.adapter.resolve() == args.output.resolve():
        raise ValueError("export output must differ from the training checkpoint")
    if (args.adapter / "adapter_config.json").is_file():
        from peft import PeftModel

        base = load_policy_model(
            args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        merged = PeftModel.from_pretrained(base, str(args.adapter)).merge_and_unload()
    elif (args.adapter / "config.json").is_file():
        merged = load_policy_model(
            str(args.adapter), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
    else:
        raise ValueError(f"not a full SFT checkpoint or LoRA adapter: {args.adapter}")
    args.output.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"save_original_format": False} if merged.config.model_type == "qwen3_5" else {}
    merged.save_pretrained(str(args.output), safe_serialization=True, **save_kwargs)
    AutoTokenizer.from_pretrained(str(args.adapter)).save_pretrained(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
