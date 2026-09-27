"""Bounded GPU replay of real tokens to compare activation recomputation costs.

No optimizer step, online sampling, reward change, or production checkpoint.
Compare all LoRA gradients, preserving FP32 IEEE and native masking.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "batch", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--max-seconds", type=int, default=3600)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    signal.signal(
        signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("probe wall budget"))
    )
    signal.alarm(args.max_seconds)
    import torch
    from omegaconf import OmegaConf
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText
    from verl.utils.qwen35_compact_head import compact_forward
    from verl.utils.qwen35_fla_ieee import install_fla_ieee, prepare_fla_ieee_runtime
    from verl.utils.qwen35_padding import install_qwen35_padding_guard

    from verl import DataProto

    os.environ.update(
        VERL_QWEN35_COMPACT_HEAD="1",
        VERL_QWEN35_COMPACT_BACKEND="checkpoint",
        VERL_QWEN35_TRIM_PADDING="experimental_both",
        VERL_QWEN35_FIX_PADDING="1",
    )
    prepare_fla_ieee_runtime()
    torch.manual_seed(42)
    base = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
    )
    install_qwen35_padding_guard(base)
    install_fla_ieee(base)
    if args.adapter:
        model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=16,
                lora_alpha=32,
                target_modules=[
                    "down_proj",
                    "gate_proj",
                    "in_proj_a",
                    "in_proj_b",
                    "in_proj_qkv",
                    "in_proj_z",
                    "k_proj",
                    "o_proj",
                    "out_proj",
                    "q_proj",
                    "up_proj",
                    "v_proj",
                ],
                bias="none",
            ),
        )
    model = model.cuda().train()
    model.enable_input_require_grads()
    if any(float(getattr(m, "p", 0)) for m in model.modules() if isinstance(m, torch.nn.Dropout)):
        raise ValueError("Numerical replay requires zero dropout")
    actor = type("ActorView", (), {})()
    actor.actor_module, actor.device_name, actor.param_dtype = model, "cuda", torch.float32
    actor.config = OmegaConf.create({"calculate_sum_pi_squared": False})
    for flag in ("use_remove_padding", "use_ulysses_sp", "use_fused_kernels", "use_prefix_grouper"):
        setattr(actor, flag, False)
    packet = DataProto.load_from_disk(args.batch)
    data = packet.batch
    # Longest observed context with actual nonzero advantage, then longest
    # context overall if distinct. Never truncate or fabricate model tokens.
    lengths = data["attention_mask"].sum(-1)
    eligible = (data["advantages"].abs() * data["response_mask"]).sum(-1) > 0
    candidates = eligible.nonzero().flatten().tolist()
    indices = [max(candidates, key=lambda i: int(lengths[i]))] if candidates else []
    longest = int(lengths.argmax())
    if longest not in indices:
        indices.append(longest)
    params = [(n, v) for n, v in model.named_parameters() if v.requires_grad]
    layers = [m for m in model.modules() if m.__class__.__name__ == "Qwen3_5DecoderLayer"]
    if not layers or not all(hasattr(m, "gradient_checkpointing") for m in layers):
        raise ValueError("Unrecognized decoder checkpoint layout")
    report = {
        "scope": "isolated LoRA weighted-logprob gradient replay, not FSDP or end-to-end RL",
        "batch": str(args.batch),
        "adapter": str(args.adapter),
        "indices": indices,
        "decoder_layers": len(layers),
        "gates": {
            "max_logprob_delta": 1e-5,
            "gradient_relative_l2": 1e-4,
            "gradient_cosine_min": 0.999999,
        },
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(args.output)

    reference = {}
    for mode in ("all", "alternating", "none"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for i, layer in enumerate(layers):
            layer.gradient_checkpointing = mode == "all" or (mode == "alternating" and i % 2 == 0)
        for index in indices:
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            micro = {
                k: data[k][index : index + 1].cuda()
                for k in (
                    "input_ids",
                    "responses",
                    "response_mask",
                    "attention_mask",
                    "position_ids",
                )
            }
            advantage = data["advantages"][index : index + 1].cuda()
            mask = micro["response_mask"]
            item = {
                "mode": mode,
                "row": index,
                "context_tokens": int(lengths[index]),
                "policy_tokens": int(mask.sum()),
                "loss": "recorded_adv_weighted_logprob",
            }
            # For an all-zero signal row, use NLL only as a numerical stress
            # probe, explicitly distinguished from a valid RL update.
            if not bool((advantage * mask).count_nonzero()):
                advantage = torch.ones_like(advantage)
                item["loss"] = "NLL_numerical_stress_only_not_RL"
            try:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.monotonic()
                out = compact_forward(actor, micro, 1.0, False, 256)["log_probs"]
                loss = -(out * advantage * mask).sum() / mask.sum().clamp_min(1)
                loss.backward()
                torch.cuda.synchronize()
                item.update(
                    seconds=time.monotonic() - start,
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                    loss_value=float(loss.detach()),
                )
                logp = out.detach().cpu()
                gradients = {
                    n: None if v.grad is None else v.grad.detach().cpu().clone() for n, v in params
                }
                finite = torch.isfinite(logp).all() and all(
                    v is None or torch.isfinite(v).all() for v in gradients.values()
                )
                if not finite:
                    raise ValueError("Nonfinite output/gradients")
                if mode == "all":
                    reference[index] = (logp, gradients)
                    item["reference"] = True
                else:
                    old_logp, old_grad = reference[index]
                    item["max_logprob_delta"] = float(((old_logp - logp) * mask.cpu()).abs().max())
                    names = [n for n, g in gradients.items() if g is not None]
                    if any((gradients[n] is None) != (old_grad[n] is None) for n in gradients):
                        raise ValueError("Gradient presence differs")
                    aa = bb = ab = dd = 0.0
                    for n in names:
                        a, b = old_grad[n].double(), gradients[n].double()
                        aa += float(a.square().sum())
                        bb += float(b.square().sum())
                        ab += float((a * b).sum())
                        dd += float((a - b).square().sum())
                    item["gradient_relative_l2"] = (dd / max(aa, 1e-30)) ** 0.5
                    item["gradient_cosine"] = ab / max((aa * bb) ** 0.5, 1e-30)
                    item["passed"] = (
                        item["max_logprob_delta"] < 1e-5
                        and item["gradient_relative_l2"] < 1e-4
                        and item["gradient_cosine"] > 0.999999
                    )
                del out, loss, gradients, logp
            except torch.cuda.OutOfMemoryError:
                item["passed"] = False
                item["error"] = "CUDA out of memory; candidate rejected"
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
            report["results"].append(item)
            save()
            print(json.dumps(item), flush=True)
            del micro, advantage, mask
            if item.get("error"):
                break
    report["completed"] = True
    save()


if __name__ == "__main__":
    main()
