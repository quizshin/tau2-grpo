"""Independent value audit for dense, unquantized Qwen3.5 TP=1 policy weights.

The audit reconstructs documented packed matrices from incoming actor tensors;
it does not call the model's weight loader to manufacture its expected values.
Vision is explicitly outside this text-only rollout, and is reported as such.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch

PACKED = {
    "q_proj": ("qkv_proj", 0, 3), "k_proj": ("qkv_proj", 1, 3),
    "v_proj": ("qkv_proj", 2, 3),
    "gate_proj": ("gate_up_proj", 0, 2), "up_proj": ("gate_up_proj", 1, 2),
    "in_proj_qkv": ("in_proj_qkvz", 0, 2), "in_proj_z": ("in_proj_qkvz", 1, 2),
    "in_proj_b": ("in_proj_ba", 0, 2), "in_proj_a": ("in_proj_ba", 1, 2),
}


def destination(name):
    if name.startswith(("model.visual.", "visual.")):
        return None
    if name.startswith("model.language_model."):
        name = "language_model.model." + name.removeprefix("model.language_model.")
    elif name.startswith("lm_head."):
        name = "language_model." + name
    if not name.startswith("language_model."):
        raise ValueError(f"Unsupported Qwen3.5 incoming parameter: {name}")
    parts = name.split(".")
    for index, part in enumerate(parts):
        if part in PACKED:
            replacement, order, count = PACKED[part]
            parts[index] = replacement
            return ".".join(parts), order, count
    return name, 0, 1


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class FullLanguageWeightAudit:
    def __init__(self):
        self.parts = {}
        self.excluded_vision = []
        self.source_names = set()

    def capture(self, weights):
        for name, value in weights:
            if name in self.source_names:
                raise ValueError(f"Duplicate incoming actor tensor: {name}")
            self.source_names.add(name)
            target = destination(name)
            if target is None:
                self.excluded_vision.append(name)
                continue
            key, order, count = target
            entries = self.parts.setdefault(key, {})
            if order in entries:
                raise ValueError(f"Duplicate packed slice: {key}/{order}")
            entries[order] = (name, count, value.detach().cpu().clone())

    def compare(self, model):
        params = dict(model.named_parameters(remove_duplicate=False))
        language = {key: value for key, value in params.items() if key.startswith("language_model.")}
        if not language:
            raise ValueError("No active language model parameters to audit")
        rows, covered = [], set()
        for key, parts in sorted(self.parts.items()):
            if key not in language:
                raise ValueError(f"Incoming weight has no active rollout destination: {key}")
            count = next(iter(parts.values()))[1]
            if set(parts) != set(range(count)) or any(p[1] != count for p in parts.values()):
                raise ValueError(f"Incomplete packed actor tensor: {key}")
            actual = language[key].detach().cpu()
            ordered = [parts[index] for index in range(count)]
            expected = (ordered[0][2] if count == 1 else torch.cat([p[2] for p in ordered], dim=0))
            expected = expected.to(actual.dtype)
            padded_rows = 0
            if actual.shape != expected.shape:
                if ".conv1d." in key and actual.numel() == expected.numel():
                    expected = expected.reshape(actual.shape)
                elif (key.endswith(("embed_tokens.weight", "lm_head.weight"))
                      and actual.ndim == expected.ndim == 2
                      and actual.shape[1] == expected.shape[1] and actual.shape[0] > expected.shape[0]):
                    padded_rows = actual.shape[0] - expected.shape[0]
                    padding = torch.zeros((padded_rows, expected.shape[1]), dtype=expected.dtype)
                    expected = torch.cat([expected, padding])
                else:
                    raise ValueError(f"Unsupported weight layout {key}: {actual.shape} vs {expected.shape}")
            match = torch.equal(actual, expected)
            rows.append({"destination": key, "sources": [p[0] for p in ordered],
                         "shape": list(actual.shape), "dtype": str(actual.dtype),
                         "exact_match": match, "vocabulary_padding_rows": padded_rows,
                         "expected_sha256": tensor_hash(expected), "actual_sha256": tensor_hash(actual)})
            covered.add(key)
        aliases = {}
        for name, parameter in language.items():
            if name in covered:
                continue
            matches = [key for key in covered if parameter.data_ptr() == language[key].data_ptr()
                       and parameter.shape == language[key].shape and parameter.stride() == language[key].stride()]
            if matches:
                aliases[name] = matches[0]
        missing = sorted(set(language) - covered - set(aliases))
        return {"schema": "tau3_qwen35_full_language_weight_audit_v1",
                "scope": "all active dense text-policy parameters; TP1, unquantized, no adapters",
                "incoming_tensor_count": len(self.source_names), "active_parameter_names": len(language),
                "compared_destinations": len(rows), "tied_aliases": aliases,
                "excluded_vision_names": sorted(self.excluded_vision),
                "missing_active_parameters": missing, "parameters": rows,
                "all_active_parameters_match": bool(rows) and not missing and all(r["exact_match"] for r in rows),
                "all_multimodal_parameters_verified": False}

    def write(self, model, output):
        report = self.compare(model)
        target = Path(output) / f"full-weight-audit-{os.getpid()}-{time.time_ns()}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n")
        if not report["all_active_parameters_match"]:
            raise RuntimeError(f"Full text-policy weight audit failed: {target}")
        return str(target)
