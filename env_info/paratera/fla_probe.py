"""Remote numerical checks for an isolated optional FLA kernel overlay."""
import copy
import json
import os
import sys

import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from peft import LoraConfig, get_peft_model
from transformers.models.qwen3_5 import modeling_qwen3_5 as native

from tau3_grpo.paths import CODE_ROOT, RESULTS_ROOT


def difference(reference, candidate):
    a, b = reference.float().flatten(), candidate.float().flatten()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    relative = ((a - b).norm() / a.norm().clamp_min(1e-12)).item()
    cosine = F.cosine_similarity(a, b, dim=0).item()
    return {"relative_l2": relative, "cosine": cosine,
            "max_absolute": (a - b).abs().max().item()}


def main():
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    report = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "kernels": []}
    for length in (63, 257, 2049):
        inputs = [torch.randn(1, length, 2, 128, device="cuda", dtype=torch.bfloat16)
                  for _ in range(3)]
        inputs += [-torch.rand(1, length, 2, device="cuda") * .2,
                   torch.rand(1, length, 2, device="cuda", dtype=torch.bfloat16)]
        results = []
        for kernel in (native.torch_chunk_gated_delta_rule, chunk_gated_delta_rule):
            values = [x.detach().clone().requires_grad_() for x in inputs]
            output, _ = kernel(*values[:3], g=values[3], beta=values[4],
                               use_qk_l2norm_in_kernel=True, output_final_state=False)
            output.float().square().mean().backward()
            results.append((output.detach(), [x.grad.detach() for x in values]))
        out = difference(results[0][0], results[1][0])
        gradients = [difference(a, b) for a, b in zip(results[0][1], results[1][1])]
        assert out["relative_l2"] < .015, out
        assert all(x["relative_l2"] < .03 and x["cosine"] > .999 for x in gradients), gradients
        report["kernels"].append({"length": length, "output": out, "gradients": gradients})
        print(json.dumps(report["kernels"][-1]), flush=True)

    sys.path.insert(0, str(CODE_ROOT / "tests"))
    from test_qwen35 import tiny_model
    from test_qwen35_loss_projection import actor

    base = tiny_model(False)
    base.model.visual.requires_grad_(False)
    fast = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0,
        target_modules=["q_proj", "in_proj_qkv"], task_type="CAUSAL_LM")).cuda()
    with torch.no_grad():
        for name, parameter in fast.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=.01)
    slow = copy.deepcopy(fast)
    found = 0
    for module in slow.modules():
        if hasattr(module, "chunk_gated_delta_rule"):
            module.chunk_gated_delta_rule = native.torch_chunk_gated_delta_rule
            found += 1
    assert found and any(getattr(x, "chunk_gated_delta_rule", None) is chunk_gated_delta_rule
                         for x in fast.modules())
    ids = torch.randint(2, 50, (2, 257), device="cuda")
    ids[0, :7] = 0
    ids[1, :11] = 0
    ids[:, -9:] = 0
    attention = ids.ne(0).long()
    mask = attention[:, 64:].clone()
    mask[:, ::3] = 0
    batch = {"input_ids": ids, "responses": ids[:, 64:], "response_mask": mask,
             "attention_mask": attention,
             "position_ids": (attention.cumsum(-1) - 1).clamp_min(0)}
    os.environ["VERL_QWEN35_LOSS_ONLY_LOGITS"] = "1"
    os.environ["VERL_QWEN35_HEAD_CHUNK_SIZE"] = "32"
    results = []
    for model in (slow, fast):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        result = actor(model, "cuda")._forward_micro_batch(batch, .7)["log_probs"]
        (-(result * mask).sum() / mask.sum()).backward()
        gradients = torch.cat([p.grad.flatten() for p in model.parameters()
                               if p.requires_grad and p.grad is not None])
        results.append((result.detach() * mask, gradients))
    report["model_output"] = difference(results[0][0], results[1][0])
    report["model_lora_gradients"] = difference(results[0][1], results[1][1])
    assert report["model_output"]["max_absolute"] < .04
    assert report["model_lora_gradients"]["relative_l2"] < .03
    assert report["model_lora_gradients"]["cosine"] > .999
    report["passed"] = True
    output = RESULTS_ROOT / "paratera-preflight/fla-numerics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
