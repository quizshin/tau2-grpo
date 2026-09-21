import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tau3_grpo.integrations.verl.cpu_saved_tensors import offloaded_update


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_native_offload_preserves_checkpointed_gradients_and_adam_state(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(42)
    initial = torch.randn(8, 8, dtype=torch.float64, device=device)
    inputs = torch.randn(7, 8, dtype=torch.float64, device=device)

    def update(weight, optimizer):
        optimizer.zero_grad(set_to_none=True)
        value = checkpoint(lambda x: (x @ weight).sin(), inputs, use_reentrant=False)
        loss = checkpoint(lambda x: (x @ weight).square().mean(), value, use_reentrant=False)
        loss.backward()
        gradient = weight.grad.detach().clone()
        optimizer.step()
        return loss.detach(), gradient

    baseline = initial.clone().requires_grad_()
    offloaded = initial.clone().requires_grad_()
    optimizers = [torch.optim.AdamW([x], lr=1e-6) for x in (baseline, offloaded)]
    for _ in range(2):
        expected = update(baseline, optimizers[0])
        actual = offloaded_update(update)(offloaded, optimizers[1])
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))
        assert torch.equal(baseline, offloaded)
        for key, value in optimizers[0].state[baseline].items():
            assert torch.equal(value, optimizers[1].state[offloaded][key])


def test_exception_restores_surrounding_saved_tensor_hooks():
    seen = []
    def pack(value):
        seen.append(value.shape)
        return value
    def fail(self):
        raise RuntimeError('deliberate')
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        with pytest.raises(RuntimeError, match='deliberate'):
            offloaded_update(fail)(None)
        x = torch.ones(3, requires_grad=True)
        (x * x).sum().backward()
    assert seen
    assert torch.equal(x.grad, torch.full_like(x, 2))
