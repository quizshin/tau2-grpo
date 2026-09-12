"""Audit the completed remote SFT export and write portable SHA256 manifests.

No training or model generation is performed. Run in the remote Qwen3.5 venv.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from importlib.metadata import version

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from tau3_grpo.models.compat import policy_model_class


ROOT = Path(os.environ.get("TAU3_ROOT", "/root/shared-nvme/tau3"))
ARTIFACT = ROOT / "artifacts/sft4b_lora_5090"
ADAPTER = ARTIFACT / "adapter"
MERGED = ARTIFACT / "sft_merged_seed42"
BASE = ROOT / "models/Qwen3.5-4B"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def indexed_weights(directory):
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        mapping = json.loads(index_path.read_text())["weight_map"]
    else:
        with safe_open(directory / "model.safetensors", framework="pt") as tensors:
            mapping = {key: "model.safetensors" for key in tensors.keys()}
    found = set()
    for filename in sorted(set(mapping.values())):
        with safe_open(directory / filename, framework="pt") as tensors:
            expected = {key for key, shard in mapping.items() if shard == filename}
            if set(tensors.keys()) != expected:
                raise RuntimeError(f"Shard index mismatch: {filename}")
            found.update(tensors.keys())
    return mapping, found


def tensor(directory, mapping, key):
    with safe_open(directory / mapping[key], framework="pt") as tensors:
        return tensors.get_tensor(key)


def main():
    torch.set_num_threads(4)
    summary = json.loads((ADAPTER / "train_summary.json").read_text())
    assert summary["actual_optimizer_steps"] == summary["expected_optimizer_steps"] == 30
    assert summary["effective_batch_size"] == 8
    assert summary["train"]["dialogues"] == 45
    assert summary["validation"]["dialogues"] == 5
    assert math.isfinite(summary["validation_metrics"]["eval_loss"])
    assert math.isclose(summary["validation_metrics"]["eval_loss"],
                        summary["best_validation_loss"], rel_tol=1e-4, abs_tol=1e-5)
    selected = Path(summary["best_model_checkpoint"])
    actual = load_file(ADAPTER / "adapter_model.safetensors")
    best = load_file(selected / "adapter_model.safetensors")
    assert actual.keys() == best.keys()
    nonzero_b = 0
    for key, value in actual.items():
        assert torch.equal(value, best[key]), f"Selected adapter mismatch: {key}"
        assert torch.isfinite(value).all(), f"Nonfinite adapter: {key}"
        if ".lora_B." in key and torch.count_nonzero(value):
            nonzero_b += 1
    assert nonzero_b > 0
    del best

    config = json.loads((ADAPTER / "adapter_config.json").read_text())
    assert not any(config.get(key) for key in
                   ("use_dora", "use_rslora", "fan_in_fan_out", "rank_pattern", "alpha_pattern"))
    scale = config["lora_alpha"] / config["r"]
    base_map, _ = indexed_weights(BASE)
    merged_map, merged_keys = indexed_weights(MERGED)
    assert not any("lora_" in key for key in merged_keys)
    assert any(key.startswith("model.language_model.layers.") for key in merged_keys)
    checked = changed = 0
    max_error = 0.0
    for key, a in actual.items():
        if not key.endswith(".lora_A.weight"):
            continue
        b = actual[key.replace(".lora_A.weight", ".lora_B.weight")]
        weight_key = key.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight")
        base = tensor(BASE, base_map, weight_key).to(torch.bfloat16)
        saved = tensor(MERGED, merged_map, weight_key)
        expected = (base.float() + (b.float() @ a.float()) * scale).to(torch.bfloat16)
        assert saved.shape == expected.shape and saved.dtype == torch.bfloat16
        assert torch.isfinite(saved).all(), weight_key
        error = (saved.float() - expected.float()).abs().max().item()
        max_error = max(max_error, error)
        # Full tensor comparison permits at most BF16 rounding-level deviation.
        torch.testing.assert_close(saved, expected, rtol=0.008, atol=1e-6, msg=weight_key)
        changed += int(not torch.equal(saved, base))
        checked += 1
    assert checked == nonzero_b and changed > 0

    for filename in sorted(set(merged_map.values())):
        with safe_open(MERGED / filename, framework="pt") as tensors:
            for key in tensors.keys():
                assert torch.isfinite(tensors.get_tensor(key)).all(), f"Nonfinite weight: {key}"
    export_config = AutoConfig.from_pretrained(MERGED, local_files_only=True)
    assert export_config.model_type == "qwen3_5"
    reloaded, loading_info = policy_model_class(export_config).from_pretrained(
        MERGED, config=export_config, torch_dtype=torch.bfloat16,
        local_files_only=True, output_loading_info=True)
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        assert not loading_info.get(field), {field: loading_info[field]}
    del reloaded
    tokenizer = AutoTokenizer.from_pretrained(MERGED, local_files_only=True)
    original = AutoTokenizer.from_pretrained(ADAPTER, local_files_only=True)
    assert tokenizer.get_vocab() == original.get_vocab()
    assert tokenizer.chat_template == original.chat_template

    code = ROOT / "code"
    config_source = code / "configs/train/sft/qwen35_4b_lora_paratera_5090.yaml"
    shutil.copy2(config_source, ARTIFACT / "sft_config.yaml")
    sources = [config_source, code / "tau3_grpo/training/sft/train.py",
               code / "tau3_grpo/training/sft/merge.py",
               ROOT / "data/sft/airline_sft_train_seed42.jsonl",
               ROOT / "data/sft/airline_sft_validation_seed42.jsonl"]
    provenance = json.loads((MERGED / "sft_provenance.json").read_text())
    provenance["source_sha256"] = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    provenance["packages"] = {name: version(name) for name in
                              ("torch", "transformers", "peft", "accelerate", "vllm")}
    (MERGED / "sft_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    report = {
        "passed": True, "optimizer_steps": 30,
        "best_checkpoint": str(selected), "adapter_tensors_match_selected": len(actual),
        "nonzero_lora_b_tensors": nonzero_b, "merged_lora_matrices_checked": checked,
        "merged_matrices_changed_from_base": changed, "merge_max_absolute_error": max_error,
        "merge_check_rtol": 0.008, "merge_check_atol": 1e-6,
        "all_merged_weights_finite": True, "merged_tensor_count": len(merged_keys),
        "full_model_reload_passed": True,
        "loading_issue_counts": {field: len(loading_info.get(field, [])) for field in
                                 ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")},
        "tokenizer_and_chat_template_match": True, "post_sft_rl_started": False,
    }
    for directory in (ADAPTER, MERGED):
        files = sorted(path for path in directory.iterdir()
                       if path.is_file() and path.name != "SHA256SUMS")
        manifest = "".join(f"{sha256(path)}  {path.name}\n" for path in files)
        (directory / "SHA256SUMS").write_text(manifest)
    (ARTIFACT / "export_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
