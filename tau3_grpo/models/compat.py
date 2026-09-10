"""Keep Qwen2.5 defaults while supporting dense Qwen3.5 checkpoints.

No GPU or Transformers import is needed for family detection. Model loading
stays lazy so data preparation and launcher validation work on a laptop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def model_family(name_or_path: str | Path) -> str:
    path = Path(name_or_path).expanduser() / "config.json"
    if path.is_file():
        kind = json.loads(path.read_text(encoding="utf-8")).get("model_type", "")
        if kind == "qwen3_5":
            return "qwen35"
        if kind.startswith("qwen3_5"):
            raise ValueError(
                "This profile supports dense Qwen3.5 conditional-generation models only"
            )
        return "legacy"
    name = str(name_or_path).lower()
    if "qwen3.5" in name:
        if "a3b" in name or "a17b" in name or "moe" in name:
            raise ValueError("The Qwen3.5 profile does not support MoE models")
        return "qwen35"
    return "legacy"


def is_qwen35_tokenizer(tokenizer: Any) -> bool:
    if model_family(getattr(tokenizer, "name_or_path", "")) == "qwen35":
        return True
    # Merged checkpoints may have arbitrary names. The native template provides
    # a second signal; the mask builder then checks its exact assistant layout.
    template = getattr(tokenizer, "chat_template", "")
    return isinstance(template, str) and "<function=" in template and "enable_thinking" in template


def chat_template_kwargs(tokenizer: Any) -> dict[str, Any]:
    return {"enable_thinking": False} if is_qwen35_tokenizer(tokenizer) else {}


def policy_model_class(config: Any) -> Any:
    from transformers import AutoModelForCausalLM

    if config.model_type == "qwen3_5":
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    if config.model_type.startswith("qwen3_5"):
        raise ValueError("Only dense Qwen3.5 checkpoints are supported by this profile")
    return AutoModelForCausalLM


def load_policy_model(name_or_path: str, **kwargs: Any) -> Any:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        name_or_path, trust_remote_code=kwargs.get("trust_remote_code", False)
    )
    model = policy_model_class(config).from_pretrained(name_or_path, config=config, **kwargs)
    if config.model_type == "qwen3_5":
        # Keep the complete HF checkpoint/weight names for vLLM and the merger,
        # but do not train the unused visual tower in a text-only experiment.
        model.model.visual.requires_grad_(False)
    return model


def lora_target_modules(model: Any, configured: str | list[str]) -> list[str]:
    if configured != "qwen35_language_linear":
        if not isinstance(configured, list) or not configured:
            raise ValueError(
                "lora.target_modules must be a nonempty list or qwen35_language_linear"
            )
        return configured
    import torch

    if model.config.model_type != "qwen3_5":
        raise ValueError("qwen35_language_linear requires a dense Qwen3.5 model")
    targets = [
        name
        for name, module in model.named_modules()
        if name.startswith("model.language_model.layers.") and isinstance(module, torch.nn.Linear)
    ]
    if not targets or not any(".linear_attn." in name for name in targets):
        raise ValueError(
            "Qwen3.5 language/Gated DeltaNet modules were not found; review model layout"
        )
    return targets


def require_training_runtime(torch: Any, family: str) -> None:
    import sys
    from importlib.metadata import version

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Training requires Python 3.12")
    if family == "qwen35":
        expected = {"torch": "2.11.0", "transformers": "5.5.1", "peft": "0.18.1"}
        for package, required in expected.items():
            actual = version(package).split("+")[0]
            if actual != required:
                raise RuntimeError(f"Qwen3.5 profile requires {package}=={required}, got {actual}")
    elif not torch.__version__.startswith("2.8") or torch.version.cuda != "12.8":
        raise RuntimeError("The legacy Qwen2.5 SFT profile requires Torch 2.8 / CUDA 12.8")
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a live CUDA GPU; local checks do not train a policy")
