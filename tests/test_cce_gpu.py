"""CCE's independent linear/log-softmax reference and first-order contract."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from areno.accel.cce import linear_selected_logprobs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _reference(x, w, labels):
    return F.linear(x, w).float().log_softmax(-1).gather(-1, labels.long()[:, None]).squeeze(-1)


def _relative_l2(actual, expected):
    return (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1, 1), (7, 33, 129), (65, 96, 2053), (128, 256, 4096)])
@pytest.mark.parametrize("trainable_head", [True, False])
def test_cce_forward_arbitrary_upstream_and_gradients(dtype, shape, trainable_head):
    n, h, v = shape
    torch.manual_seed(42)
    x = (torch.randn(n, h, device="cuda", dtype=dtype) / h**0.5).requires_grad_()
    w = torch.randn(v, h, device="cuda", dtype=dtype).requires_grad_(trainable_head)
    xr = x.detach().clone().requires_grad_()
    wr = w.detach().clone().requires_grad_(trainable_head)
    labels = torch.randint(v, (n,), device="cuda")
    labels[0] = v - 1
    upstream = torch.randn(n, device="cuda")
    upstream[::3] = 0  # masking is an arbitrary upstream zero, not a different loss
    actual = linear_selected_logprobs(x, w, labels)
    expected = _reference(xr, wr, labels)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    actual.backward(upstream)
    expected.backward(upstream)
    limit = 1e-4 if dtype == torch.float32 else 0.02
    assert _relative_l2(x.grad, xr.grad) <= limit
    if trainable_head:
        assert _relative_l2(w.grad, wr.grad) <= limit
    else:
        assert w.grad is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cce_noncontiguous_tied_weight_and_reduction(dtype):
    torch.manual_seed(11)
    w = torch.randn(257, 64, device="cuda", dtype=dtype).T.contiguous().T.requires_grad_()
    wr = w.detach().clone().requires_grad_()
    ids = torch.tensor([1, 5, 1, 256], device="cuda")
    labels = torch.tensor([256, 0, 2, 1], device="cuda", dtype=torch.int32)
    x = F.embedding(ids, w) / 8
    xr = F.embedding(ids, wr) / 8
    actual = linear_selected_logprobs(x, w, labels)
    expected = _reference(xr, wr, labels)
    (-actual.mean()).backward()
    (-expected.mean()).backward()
    limit = 1e-4 if dtype == torch.float32 else 0.02
    assert _relative_l2(w.grad, wr.grad) <= limit
    torch.testing.assert_close(actual, expected, atol=limit, rtol=limit)


def test_cce_empty_and_frozen_hidden():
    w = torch.randn(129, 33, device="cuda", requires_grad=True)
    x = torch.empty(0, 33, device="cuda", requires_grad=True)
    linear_selected_logprobs(x, w, torch.empty(0, device="cuda", dtype=torch.long)).sum().backward()
    assert x.grad.shape == x.shape
    assert torch.count_nonzero(w.grad) == 0
    w.grad = None
    x = torch.randn(7, 33, device="cuda")
    labels = torch.arange(7, device="cuda")
    actual = linear_selected_logprobs(x, w, labels)
    actual.sum().backward()
    wr = w.detach().clone().requires_grad_()
    _reference(x, wr, labels).sum().backward()
    assert _relative_l2(w.grad, wr.grad) <= 1e-4


def test_cce_cuda_graph_capture_replay():
    x = torch.randn(33, 64, device="cuda", requires_grad=True)
    w = torch.randn(257, 64, device="cuda", requires_grad=True)
    labels = torch.arange(33, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            x.grad = w.grad = None
            linear_selected_logprobs(x, w, labels).sum().backward()
    torch.cuda.current_stream().wait_stream(stream)
    x.grad = w.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = linear_selected_logprobs(x, w, labels)
        result.sum().backward()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(result, _reference(x, w, labels), atol=1e-5, rtol=1e-5)
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(w.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cce_noncontiguous_hidden_large_logits_and_zero_mask(dtype):
    torch.manual_seed(9)
    x = (torch.randn(64, 35, device="cuda", dtype=dtype) * 2).T.contiguous().T.requires_grad_()
    w = torch.randn(131, 35, device="cuda", dtype=dtype).requires_grad_()
    xr = x.detach().clone().requires_grad_()
    wr = w.detach().clone().requires_grad_()
    labels = torch.arange(64, device="cuda")
    actual, expected = linear_selected_logprobs(x, w, labels), _reference(xr, wr, labels)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    actual.backward(torch.zeros_like(actual))
    assert torch.count_nonzero(x.grad) == torch.count_nonzero(w.grad) == 0


def test_cce_packed_gpu_excludes_sequence_tails():
    from areno.engine.runtime.logprobs import packed_cce_logprobs

    torch.manual_seed(42)
    hidden = torch.randn(1, 7, 33, device="cuda", requires_grad=True)
    weight = torch.randn(129, 33, device="cuda", requires_grad=True)
    tokens = torch.tensor([[1, 128, 2, 8, 5, 9, 3]], device="cuda")
    cu = torch.tensor([0, 3, 4, 7], device="cuda", dtype=torch.int32)
    positions = torch.tensor([0, 1, 4, 5], device="cuda")
    result = packed_cce_logprobs(hidden, tokens, cu, weight)
    expected = _reference(hidden[0, positions], weight, tokens[0, positions + 1])
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
    actual_grads = torch.autograd.grad(result.sum(), (hidden, weight), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (hidden, weight))
    for actual, reference in zip(actual_grads, expected_grads, strict=True):
        assert _relative_l2(actual, reference) <= 1e-4
