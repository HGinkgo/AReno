from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context
from areno.models.qwen3.model import Qwen3Adapter


@pytest.fixture(autouse=True)
def _cpu_tp_context(monkeypatch):
    monkeypatch.setenv("ARENO_CKPT_PROGRESS", "0")
    previous = get_tp_context()
    set_tp_context(TPContext(rank=0, world_size=1, device=torch.device("cpu"), group=None))
    try:
        yield
    finally:
        set_tp_context(previous)


def _checkpoint(tmp_path, tied):
    config = {
        "model_type": "qwen3",
        "vocab_size": 8,
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 0,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "tie_word_embeddings": tied,
        "dtype": "float32",
    }
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config))
    tensors = {
        "model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32,
        "model.norm.weight": torch.ones(4),
    }
    if not tied:
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].flip(0)
    save_file(tensors, source / "model.safetensors")
    adapter = Qwen3Adapter()
    return source, adapter, adapter.config_from_hf(config)


def _logits(model):
    # Exercise both parameter uses without requiring CUDA embedding kernels.
    return F.linear(F.embedding(torch.tensor([1, 3, 5]), model.embed_tokens.weight), model.lm_head.weight)


@pytest.mark.parametrize("tied", [True, False])
def test_training_update_survives_checkpoint_roundtrip(tmp_path, tied):
    source, adapter, config = _checkpoint(tmp_path, tied)
    model = adapter.build(config)
    adapter.load_weights(model, source)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    F.cross_entropy(_logits(model), torch.tensor([2, 4, 6])).backward()
    optimizer.step()
    expected = _logits(model).detach()

    saved = tmp_path / "saved"
    adapter.save_weights(model, saved, source)
    restored = adapter.build(config)
    adapter.load_weights(restored, saved)

    torch.testing.assert_close(_logits(restored), expected, rtol=0, atol=0)
    assert (restored.lm_head.weight is restored.embed_tokens.weight) is tied
    weight_map = json.loads((saved / "model.safetensors.index.json").read_text())["weight_map"]
    assert ("lm_head.weight" in weight_map) is not tied


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_tied_checkpoint_shards_share_parameter_and_accumulate_both_gradients(tmp_path, tp_rank):
    source, adapter, config = _checkpoint(tmp_path, True)
    set_tp_context(TPContext(rank=tp_rank, world_size=2, device=torch.device("cpu"), group=None))
    model = adapter.build(config)
    adapter.load_weights(model, source)

    assert model.lm_head.weight is model.embed_tokens.weight
    assert sum(parameter.numel() for parameter in model.parameters()) == 4 * 4 + 4
    shared = model.embed_tokens.weight
    reference = shared.detach().clone().requires_grad_()
    tokens = torch.tensor([0, 2])
    F.linear(F.embedding(tokens, shared), model.lm_head.weight).square().sum().backward()
    F.linear(F.embedding(tokens, reference), reference).square().sum().backward()
    torch.testing.assert_close(shared.grad, reference.grad, rtol=0, atol=0)
