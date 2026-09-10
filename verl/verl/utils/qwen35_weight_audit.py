"""Opt-in, local diagnostics for Qwen3.5 rollout base and adapter weights."""

import hashlib
import json
import os
import re
import time
from pathlib import Path

import torch
from safetensors import safe_open


def audit_qwen35_rollout_weights(model, checkpoint, output_dir, phase):
    """Compare unsharded GDN conv weights and fingerprint actual GPU LoRA buffers.

    This is intended for short TP=1 integration checks, not routine training.
    It runs inside the trusted worker after a complete weight transfer; no
    remote callable deserialization or model weight dumps are required.
    """
    params = dict(model.named_parameters())
    conv = {}
    for key, value in params.items():
        if ".conv1d." in key and key.endswith(".weight"):
            native = "model.language_model." + re.search(r"layers\..*", key).group().replace(".base_layer.", ".")
            conv[native] = (key, value.detach().cpu())
    report = {"time": time.time(), "phase": phase, "conv": [], "lora_gpu": []}
    for shard in sorted(Path(checkpoint).glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as saved:
            for native in set(saved.keys()) & conv.keys():
                key, tensor = conv[native]
                expected = saved.get_tensor(native).to(tensor.dtype)
                if expected.numel() == tensor.numel():
                    expected = expected.reshape(tensor.shape)
                matched = torch.equal(tensor, expected)
                report["conv"].append({"key": key, "shape": list(tensor.shape), "exact_match": matched})
    for name, module in model.named_modules():
        for attr in ("lora_a_stacked", "lora_b_stacked"):
            stack = getattr(module, attr, None)
            if stack is None:
                continue
            tensors = stack if isinstance(stack, (tuple, list)) else [stack]
            for index, value in enumerate(tensors):
                if not isinstance(value, torch.Tensor):
                    continue
                tensor = value.detach().cpu().contiguous()
                report["lora_gpu"].append(
                    {
                        "key": f"{name}.{attr}[{index}]",
                        "shape": list(tensor.shape),
                        "nonzero": int(torch.count_nonzero(tensor)),
                        "sha256": hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest(),
                    }
                )
    report["all_conv_match"] = (
        bool(conv) and len(report["conv"]) == len(conv) and all(x["exact_match"] for x in report["conv"])
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"worker-audit-{os.getpid()}-{time.time_ns()}.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_conv_match"]:
        raise RuntimeError(f"Qwen3.5 rollout convolution weights differ from the base checkpoint: {target}")
    return str(target)
