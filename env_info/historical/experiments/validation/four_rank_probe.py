"""One real-model PPO actor update on a complete dialogue padded to 8K + 16K.

Diagnostic only: synthetic advantages, no rollout/reward claim, no saved weights.
Exercises old/ref scoring and the production actor forward with FSDP and LoRA.
"""
import datetime
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from peft import LoraConfig, get_peft_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from transformers import AutoTokenizer, set_seed

from tau3_grpo.models.compat import load_policy_model, lora_target_modules, require_training_runtime
from tau3_grpo.paths import CODE_ROOT, CONFIG_ROOT, MODEL_ROOT, RESULTS_ROOT, SFT_DATA_ROOT
from tau3_grpo.training.sft.dataset import TrajectorySFTDataset
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, resolve_fsdp_use_orig_params


def main():
    sys.path.insert(0, str(CODE_ROOT / "tests"))
    from test_qwen35_loss_projection import actor

    rank=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    require_training_runtime(torch, "qwen35")
    set_seed(42)
    os.environ["VERL_QWEN35_LOSS_ONLY_LOGITS"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT / "Qwen3.5-4B", local_files_only=True)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tools = [x["tool_schema"] for x in yaml.safe_load(
        (CONFIG_ROOT / "envs/tool_config.yaml").read_text())["tools"]]
    candidates = []
    for split, size in (("train", 45), ("validation", 5)):
        dataset = TrajectorySFTDataset(SFT_DATA_ROOT / f"airline_sft_{split}_seed42.jsonl",
            tokenizer, tools=tools, max_length=24576, expected_size=size)
        for index, example in enumerate(dataset.examples):
            first = next(i for i, label in enumerate(example["labels"]) if label != -100)
            if 0 < first <= 8192 and len(example["input_ids"]) - first <= 16384:
                candidates.append((example["n_label_tokens"], split, index, first, example))
    _, split, index, first, example = max(candidates, key=lambda x: x[0])
    ids = example["input_ids"]
    left, right = 8192 - first, 16384 - (len(ids) - first)
    batch = {
        "input_ids": torch.tensor([[pad] * left + ids + [pad] * right], device="cuda"),
        "responses": torch.tensor([ids[first:] + [pad] * right], device="cuda"),
        "attention_mask": torch.tensor([[0] * left + [1] * len(ids) + [0] * right], device="cuda"),
        "response_mask": torch.tensor([[int(x != -100) for x in example["labels"][first:]]
                                       + [0] * right], device="cuda"),
    }
    batch["position_ids"] = (batch["attention_mask"].cumsum(-1) - 1).clamp_min(0)
    report = {"kind": "synthetic_advantage_diagnostic", "split": split, "index": index,
              "original_tokens": len(ids), "padded_tokens": 24576,
              "policy_tokens": example["n_label_tokens"], "gpu": torch.cuda.get_device_name()}
    print(json.dumps(report), flush=True)
    base = load_policy_model(str(MODEL_ROOT / "Qwen3.5-4B"), torch_dtype=torch.bfloat16,
                             attn_implementation="sdpa", trust_remote_code=False)
    # Lazy backend imports may consume RNG during base construction. Reset at
    # adapter initialization so a kernel comparison starts from identical LoRA.
    set_seed(42)
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=lora_target_modules(base, "qwen35_language_linear")))
    fingerprint = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            fingerprint.update(name.encode())
            fingerprint.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    report["initial_adapter_sha256"] = fingerprint.hexdigest()
    print(json.dumps({"initial_adapter_sha256": fingerprint.hexdigest()}), flush=True)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    with tempfile.TemporaryDirectory() as tmp:
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=180))
        try:
            wrapped = FSDP(model, device_id=torch.device("cuda", rank),
                auto_wrap_policy=get_fsdp_wrap_policy(model, is_lora=True),
                use_orig_params=resolve_fsdp_use_orig_params(model, True, True),
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32, buffer_dtype=torch.float32))
            worker = actor(wrapped, "cuda")
            optimizer = torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad], lr=1e-6)
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            wrapped.eval()
            with torch.no_grad():
                old = worker._forward_micro_batch(batch, 1.0, calculate_entropy=True)
                with wrapped.disable_adapter():
                    ref = worker._forward_micro_batch(batch, 1.0)["log_probs"]
            report["scoring_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            del old
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            wrapped.train()
            logp = worker._forward_micro_batch(batch, 1.0)["log_probs"]
            mask = batch["response_mask"]
            signs = torch.where(torch.arange(16384, device="cuda") % 2 == 0, 1., -1.)
            loss = ((-logp * signs + .01 * (logp - ref).square()) * mask).sum() / mask.sum()
            loss.backward()
            norm = wrapped.clip_grad_norm_(1.0)
            assert torch.isfinite(loss) and torch.isfinite(norm) and norm > 0
            tracked = next(p for p in wrapped.parameters()
                           if p.requires_grad and p.grad is not None and p.grad.count_nonzero())
            before = tracked.detach().cpu().clone()
            assert torch.equal(before, tracked.detach().cpu()), "unexpected parameter change"
            torch.cuda.synchronize()
            report.update(loss=loss.item(), grad_norm=norm.item(), optimizer_change=False, rank=rank, world_size=dist.get_world_size(),
                update_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                update_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                elapsed_seconds=time.monotonic() - start)
            # Optional remote-only numerical audit between native and FLA kernels.
            # Save gradients and scored tokens, not model/checkpoint weights.
            snapshot = os.environ.get("TAU3_PROBE_COMPARE_FILE")
            if snapshot:
                target = Path(snapshot)
                target.parent.mkdir(parents=True, exist_ok=True)
                parameters = [(name, p) for name, p in wrapped.named_parameters()
                              if p.requires_grad and p.grad is not None]
                torch.save({"names": [name for name, _ in parameters],
                    "gradients": [p.grad.detach().cpu() for _, p in parameters],
                    "log_probs": logp.detach()[mask.bool()].cpu()}, target)
                report["comparison_snapshot"] = str(target)
            output = Path("/root/autodl-tmp/tau3-5xa800-20260911/validation/four-rank-preflight")
            output.mkdir(exist_ok=True, parents=True)
            (output / f"rl-memory-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
