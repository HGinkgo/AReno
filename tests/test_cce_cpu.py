from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from areno.accel.cce import linear_selected_logprobs
from areno.engine import training
from areno.engine.layers.vocab import VocabParallelLMHead
from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context
from areno.engine.runtime.logprobs import cce_head_supported, packed_cce_logprobs


def _worker(*, compile_model, model_type="qwen3"):
    return SimpleNamespace(
        config=SimpleNamespace(
            runtime=SimpleNamespace(compile_model=compile_model),
            model=SimpleNamespace(model_type=model_type),
        )
    )


def test_cce_cpu_and_modified_head_are_not_dispatched():
    previous = get_tp_context()
    set_tp_context(TPContext(rank=0, world_size=1, device=torch.device("cpu"), group=None))
    try:
        assert not cce_head_supported(VocabParallelLMHead(4, 8))
        assert not cce_head_supported(torch.nn.Linear(4, 8))
        with pytest.raises(ValueError, match="CUDA"):
            linear_selected_logprobs(torch.zeros(2, 4), torch.zeros(8, 4), torch.zeros(2, dtype=torch.long))
    finally:
        set_tp_context(previous)


@pytest.mark.parametrize("lengths", [[3, 1, 2], [1, 1], [4], [1, 4, 1, 2]])
def test_packed_cce_preserves_boundaries_and_masks(monkeypatch, lengths):
    recorded = {}

    def capture(hidden, weight, labels):
        recorded.update(hidden=hidden, labels=labels)
        return hidden[:, 0]

    monkeypatch.setattr("areno.accel.cce.linear_selected_logprobs", capture)
    tokens = torch.arange(sum(lengths)).view(1, -1)
    hidden = tokens.float().unsqueeze(-1).requires_grad_()
    cu = torch.tensor([0] + lengths).cumsum(0).to(torch.int32)
    selected = packed_cce_logprobs(hidden, tokens, cu, torch.empty(0))
    tails = {int(value) - 1 for value in cu[1:]}
    positions = [index for index in range(tokens.numel()) if index not in tails]
    torch.testing.assert_close(recorded["labels"], torch.tensor([i + 1 for i in positions], dtype=torch.long))
    torch.testing.assert_close(selected, torch.tensor(positions, dtype=torch.float32))
    selected.sum().backward()
    expected = torch.ones_like(hidden)
    expected[:, list(tails)] = 0
    torch.testing.assert_close(hidden.grad, expected)


@pytest.mark.parametrize(
    ("env", "compile_model", "model_type", "world_size", "data_pack", "head_supported", "expected"),
    [
        ("1", False, "qwen3", 1, {"train_cu_seqlens": object()}, True, True),
        ("0", False, "qwen3", 1, {"train_cu_seqlens": object()}, True, False),
        ("1", True, "qwen3", 1, {"train_cu_seqlens": object()}, True, False),
        ("1", False, "llama", 1, {"train_cu_seqlens": object()}, True, False),
        ("1", False, "qwen3", 2, {"train_cu_seqlens": object()}, True, False),
        ("1", False, "qwen3", 1, {}, True, False),
        ("1", False, "qwen3", 1, {"train_cu_seqlens": object()}, False, False),
    ],
)
def test_cce_dispatch_requires_all_supported_conditions(
    monkeypatch, env, compile_model, model_type, world_size, data_pack, head_supported, expected
):
    monkeypatch.setenv("ARENO_CCE", env)
    monkeypatch.setattr(training, "cce_head_supported", lambda head: head_supported)
    ctx = SimpleNamespace(world_size=world_size)
    train_model = SimpleNamespace(lm_head=object())

    assert (
        training._cce_training_supported(
            _worker(compile_model=compile_model, model_type=model_type), ctx, data_pack, train_model
        )
        is expected
    )
