"""Long reference inference must not allocate a full sequence softmax copy."""
import pytest
import torch
import torch.nn.functional as F
from verl.utils.torch_functional import logprobs_from_logits_v2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_no_grad_low_precision_logprobs_are_exact_and_chunked(dtype, noncontiguous, monkeypatch):
    torch.manual_seed(42)
    logits = torch.randn(2, 521, 194, dtype=dtype)
    if noncontiguous:
        logits = logits[..., ::2]
    labels = torch.randint(logits.shape[-1], logits.shape[:-1])
    expected = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    original = F.log_softmax
    shapes = []

    def record(tensor, *args, **kwargs):
        shapes.append(tensor.shape)
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(F, "log_softmax", record)
    with torch.no_grad():
        actual = logprobs_from_logits_v2(logits, labels)
    assert torch.equal(actual, expected)
    assert actual.dtype == dtype
    assert len(shapes) == 10
    assert all(shape[0] <= 128 and shape[-1] == logits.shape[-1] for shape in shapes)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_logprob_training_gradient_is_preserved(dtype):
    torch.manual_seed(42)
    logits = torch.randn(2, 9, 33, dtype=dtype, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    labels = torch.randint(33, (2, 9))
    actual = logprobs_from_logits_v2(logits, labels)
    expected = F.log_softmax(reference, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    actual.sum().backward()
    expected.sum().backward()
    tolerance = 1e-6 if dtype == torch.float32 else 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=tolerance)
    torch.testing.assert_close(logits.grad, reference.grad, rtol=0, atol=tolerance)


def test_two_dimensional_logits_keep_vocabulary_axis_intact():
    logits = torch.randn(3, 521, dtype=torch.bfloat16)
    labels = torch.tensor([0, 129, 520])
    with torch.no_grad():
        expected = F.log_softmax(logits, dim=-1).gather(-1, labels[:, None]).squeeze(-1)
        assert torch.equal(logprobs_from_logits_v2(logits, labels), expected)
