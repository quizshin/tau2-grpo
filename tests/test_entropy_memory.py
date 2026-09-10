"""No-grad long-sequence entropy preserves numerics without a full softmax copy."""

import pytest
import torch
import torch.nn.functional as F
from verl.utils.torch_functional import entropy_from_logits


def original_entropy(logits):
    pd = F.softmax(logits, dim=-1)
    return torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_entropy_chunking_matches_original_and_bounds_temporaries(
    dtype, noncontiguous, monkeypatch
):
    torch.manual_seed(7)
    logits = torch.randn(2, 521, 194, dtype=dtype)
    if noncontiguous:
        logits = logits[..., ::2]
    expected = original_entropy(logits)
    real_softmax = F.softmax
    shapes = []

    def record(value, *args, **kwargs):
        shapes.append(value.shape)
        return real_softmax(value, *args, **kwargs)

    monkeypatch.setattr(F, "softmax", record)
    with torch.no_grad():
        result = entropy_from_logits(logits)
    assert torch.equal(result, expected)
    assert len(shapes) == 5 and all(shape[1] <= 128 for shape in shapes)
    assert result.dtype == expected.dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_entropy_autograd_path_preserves_gradients(dtype):
    torch.manual_seed(9)
    logits = torch.randn(2, 257, 31, dtype=dtype, requires_grad=True)
    other = logits.detach().clone().requires_grad_()
    actual = entropy_from_logits(logits)
    expected = original_entropy(other)
    actual.sum().backward()
    expected.sum().backward()
    assert torch.equal(actual, expected)
    assert torch.equal(logits.grad, other.grad)


@pytest.mark.parametrize("shape", [(2, 0, 33), (0, 257, 33), (257, 33), (2, 128, 33)])
def test_entropy_other_shapes_remain_valid(shape):
    logits = torch.randn(*shape)
    with torch.no_grad():
        assert torch.equal(entropy_from_logits(logits), original_entropy(logits))
