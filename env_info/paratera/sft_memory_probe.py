"""Remote 4B LoRA preflight on the longest frozen SFT examples.

Uses real full dialogues, native masking and the production supervised loss.
Diagnostic updates use distinct worst-case examples and are discarded;
no model is saved or used for RL.
"""
import gc
import json
import time

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, set_seed

from tau3_grpo.models.compat import load_policy_model, lora_target_modules, require_training_runtime
from tau3_grpo.paths import CONFIG_ROOT, MODEL_ROOT, RESULTS_ROOT, SFT_DATA_ROOT
from tau3_grpo.training.sft.dataset import TrajectorySFTDataset, collate_fn_padding
from tau3_grpo.training.sft.trainer import Qwen35SFTTrainer


def main():
    require_training_runtime(torch, "qwen35")
    set_seed(42)
    model_path = MODEL_ROOT / "Qwen3.5-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tools = [x["tool_schema"] for x in yaml.safe_load(
        (CONFIG_ROOT / "envs/tool_config.yaml").read_text())["tools"]]
    datasets = {split: TrajectorySFTDataset(
        SFT_DATA_ROOT / f"airline_sft_{split}_seed42.jsonl", tokenizer, tools=tools,
        max_length=24576, expected_size=45 if split == "train" else 5)
        for split in ("train", "validation")}
    examples = [(split, i, x) for split, dataset in datasets.items()
                for i, x in enumerate(dataset.examples)]
    selected = {}
    for key in ("n_total_tokens", "n_label_tokens"):
        split, index, example = max(examples, key=lambda x: x[2][key])
        selected[(split, index)] = example
    report = {"kind": "diagnostic_only", "gpu": torch.cuda.get_device_name(),
              "dataset_stats": {k: v.token_stats() for k, v in datasets.items()}, "probes": []}
    print(json.dumps(report), flush=True)
    base = load_policy_model(str(model_path), torch_dtype=torch.bfloat16,
                             attn_implementation="sdpa", trust_remote_code=False)
    model = get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=lora_target_modules(base, "qwen35_language_linear")))
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.cuda().train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0)
    for (split, index), example in selected.items():
        batch = {k: v.cuda() for k, v in collate_fn_padding(
            [datasets[split][index]], tokenizer.pad_token_id).items()}
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Production compute_loss is stateless and does not require a Trainer instance.
            loss = Qwen35SFTTrainer.compute_loss(None, model, batch)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        assert torch.isfinite(loss) and torch.isfinite(norm) and norm > 0
        changed = next(p for p in trainable if p.grad is not None and p.grad.count_nonzero())
        before = changed.detach().cpu().clone()
        optimizer.step()
        assert not torch.equal(before, changed.detach().cpu()), "optimizer did not change an adapter"
        torch.cuda.synchronize()
        result = {"split": split, "index": index,
                  "tokens": example["n_total_tokens"], "labels": example["n_label_tokens"],
                  "loss": loss.item(), "grad_norm": norm.item(),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "elapsed_seconds": time.monotonic() - start}
        report["probes"].append(result)
        print(json.dumps(result), flush=True)
        del loss, batch, before
    output = RESULTS_ROOT / "paratera-preflight"
    output.mkdir(parents=True, exist_ok=True)
    (output / "sft-memory.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
