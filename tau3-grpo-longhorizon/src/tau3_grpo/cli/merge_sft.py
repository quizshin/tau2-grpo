"""Merge the LoRA SFT adapter into a standalone Hugging Face checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge the Airline SFT LoRA adapter")
    parser.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("checkpoint merge requires Python 3.12")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.__version__.startswith("2.8"):
        raise RuntimeError(f"checkpoint merge requires torch 2.8, got {torch.__version__}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    merged = PeftModel.from_pretrained(base, str(args.adapter)).merge_and_unload()
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(args.adapter)).save_pretrained(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
