"""Run remotely with torchrun --standalone --nproc-per-node=2.

Exercises CUDA BF16 forward/backward and real NCCL collectives on both cards.
This is an environment check, not evidence that the full training model fits.
"""
import datetime
import importlib.metadata
import json
import os

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    expected = {"torch": "2.11.0", "vllm": "0.20.0", "transformers": "5.5.1",
                "peft": "0.18.1", "accelerate": "1.12.0"}
    versions = {name: importlib.metadata.version(name) for name in expected}
    for name, version in versions.items():
        assert version.split("+")[0] == expected[name], (name, version)
    # Require the installed CUDA extension to load.
    import vllm._C  # noqa: F401
    from tau3_grpo.models.compat import require_training_runtime
    require_training_runtime(torch, "qwen35")
    assert torch.cuda.get_device_capability(rank) == (12, 0)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90),
                            device_id=torch.device("cuda", rank))
    try:
        x = torch.eye(256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y = x @ x
        torch.testing.assert_close(y, x)
        y.float().sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 2))
        value = torch.tensor([rank + 1.0], device="cuda")
        dist.all_reduce(value)
        size = dist.get_world_size()
        assert value.item() == size * (size + 1) / 2
        # Average local gradients from a small differentiable objective.
        p = torch.tensor([2.0], device="cuda", requires_grad=True)
        (p * (rank + 1)).square().sum().backward()
        dist.all_reduce(p.grad)
        p.grad.div_(size)
        expected_grad = 4 * sum(i * i for i in range(1, size + 1)) / size
        assert p.grad.item() == expected_grad
        torch.cuda.synchronize()
        print(json.dumps({"rank": rank, "gpu": torch.cuda.get_device_name(rank),
                          "cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version(),
                          "versions": versions, "bf16_backward": "passed",
                          "nccl_allreduce": value.item(), "averaged_gradient": p.grad.item(),
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(rank)}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
