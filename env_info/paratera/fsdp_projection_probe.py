"""Two-rank GPU regression for masked Qwen3.5 LoRA projection and accumulation."""
import copy
import datetime
import json
import os
import sys

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision

from tau3_grpo.paths import CODE_ROOT
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, resolve_fsdp_use_orig_params

def main():
    sys.path.insert(0, str(CODE_ROOT / "tests"))
    from test_qwen35 import tiny_model
    from test_qwen35_loss_projection import actor

    rank = int(os.environ["LOCAL_RANK"])
    os.environ["VERL_QWEN35_HEAD_CHUNK_SIZE"] = "2"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank),
                            timeout=datetime.timedelta(seconds=90))
    try:
        torch.manual_seed(42)
        base = tiny_model(True).to(torch.bfloat16)
        base.model.visual.requires_grad_(False)
        model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0,
            target_modules=["q_proj", "in_proj_qkv"], task_type="CAUSAL_LM"))
        copies = [model, copy.deepcopy(model)]
        wrapped = []
        for candidate in copies:
            candidate.enable_input_require_grads()
            candidate.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            wrapped.append(FSDP(candidate, device_id=torch.device("cuda", rank),
                auto_wrap_policy=get_fsdp_wrap_policy(candidate, is_lora=True),
                use_orig_params=resolve_fsdp_use_orig_params(candidate, True, True),
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32, buffer_dtype=torch.float32)))
        gradients = []
        for projected, candidate in enumerate(wrapped):
            os.environ["VERL_QWEN35_LOSS_ONLY_LOGITS"] = str(projected)
            candidate.train()
            for micro in range(2):
                ids = torch.tensor([[0, 2, 3, 4, 5, 6, 7, 8, 0, 0]], device="cuda")
                attention = ids.ne(0).long()
                mask = torch.tensor([[1, 0, 1, 1, 0, 0]], device="cuda")
                if rank == micro:
                    mask.zero_()  # Different ranks are empty in the two micro-batches.
                batch = {"input_ids": ids, "responses": ids[:, 4:], "attention_mask": attention,
                         "position_ids": (attention.cumsum(-1) - 1).clamp_min(0),
                         "response_mask": mask}
                out = actor(candidate, "cuda")._forward_micro_batch(batch, 1.0)
                (-(out["log_probs"] * mask).sum() / 2).backward()
            norm = candidate.clip_grad_norm_(1.0)
            assert torch.isfinite(norm) and norm > 0
            params = [p for p in candidate.parameters() if p.requires_grad]
            gradients.append([p.grad.detach().clone() if p.grad is not None else None for p in params])
            before = [p.detach().clone() for p in params]
            torch.optim.AdamW(params, lr=1e-3).step()
            changed = torch.tensor(int(any(not torch.equal(a, b) for a, b in zip(before, params))), device="cuda")
            dist.all_reduce(changed)
            assert changed > 0
        for expected, actual in zip(*gradients):
            assert (expected is None) == (actual is None)
            if expected is not None:
                torch.testing.assert_close(actual, expected, atol=.01, rtol=.04)
        print(json.dumps({"rank": rank, "world_size": dist.get_world_size(),
                          "fsdp_projection_gradients": "passed", "micro_batches": 2,
                          "empty_rank_masks": "passed", "optimizer_change": "passed"}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
