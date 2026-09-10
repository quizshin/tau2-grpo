"""Fetch revision-pinned policy/tokenizer assets; never called by training."""

import argparse
import json
from pathlib import Path

from tau3_grpo.paths import CONFIG_ROOT, MODEL_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=["0.8B", "2B", "4B", "9B"], default="0.8B")
    parser.add_argument(
        "--include-user", action="store_true", help="Also fetch the Qwen3.5-4B simulator"
    )
    parser.add_argument(
        "--include-legacy-tokenizer",
        action="store_true",
        help="Fetch only Qwen2.5 tokenizer/config assets for the frozen data split and regressions",
    )
    parser.add_argument(
        "--tokenizer-only", action="store_true", help="No policy weights; useful for local tests"
    )
    parser.add_argument("--output-root", type=Path, default=MODEL_ROOT)
    args = parser.parse_args(argv)
    from huggingface_hub import snapshot_download

    pins = json.loads((CONFIG_ROOT / "models/qwen35.json").read_text(encoding="utf-8"))
    models = [f"Qwen/Qwen3.5-{args.size}"]
    if args.include_user:
        models.append("Qwen/Qwen3.5-4B")
    if args.include_legacy_tokenizer:
        models.append("Qwen/Qwen2.5-7B-Instruct")
    tokenizer_patterns = [
        "config.json",
        "tokenizer*",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    ]
    for model in dict.fromkeys(models):
        tokenizer_only = args.tokenizer_only or model == "Qwen/Qwen2.5-7B-Instruct"
        patterns = (
            tokenizer_patterns
            if tokenizer_only
            else tokenizer_patterns
            + [
                "*.safetensors",
                "*.safetensors.index.json",
                "preprocessor_config.json",
                "video_preprocessor_config.json",
            ]
        )
        target = args.output_root / model.split("/")[-1]
        snapshot_download(model, revision=pins[model], local_dir=target, allow_patterns=patterns)
        (target / "tau3_source_revision.json").write_text(
            json.dumps(
                {"model": model, "revision": pins[model], "tokenizer_only": tokenizer_only},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
