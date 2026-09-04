from __future__ import annotations

import torch

from tiny_deepseek.core.model import build_model
from tests.test_model import tiny_config
from tiny_deepseek.core.utils import load_checkpoint, save_checkpoint


def test_checkpoint_restores_identical_greedy_logits(tmp_path) -> None:
    torch.manual_seed(5)
    model = build_model(tiny_config(router_type="gru")).eval()
    ids = torch.randint(0, model.config.vocab_size, (2, model.config.context_length))
    expected = model(ids, routing_mode="greedy").logits.detach()
    path = tmp_path / "model.pt"
    save_checkpoint(
        path=path, model=model, optimizer=None, scheduler=None, step=7,
        stoi={str(i): i for i in range(model.config.vocab_size)},
        itos={i: str(i) for i in range(model.config.vocab_size)},
        training_config={}, best_metrics={},
    )
    restored, checkpoint = load_checkpoint(path, "cpu")
    actual = restored.eval()(ids, routing_mode="greedy").logits.detach()
    assert checkpoint["step"] == 7
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_hybrid_checkpoint_restores_both_router_levels(tmp_path) -> None:
    torch.manual_seed(15)
    config = tiny_config(
        model_type="mor_skip", router_type="linear", n_layers=8,
        recursion_steps=3, recursion_block_layers=2,
        router_bias=True, initial_execute_probability=0.9,
    )
    model = build_model(config).eval()
    ids = torch.randint(0, config.vocab_size, (2, config.context_length))
    expected = model(ids, routing_mode="greedy")
    path = tmp_path / "hybrid.pt"
    save_checkpoint(
        path=path, model=model, optimizer=None, scheduler=None, step=3,
        stoi={str(i): i for i in range(config.vocab_size)},
        itos={i: str(i) for i in range(config.vocab_size)},
        training_config={}, best_metrics={},
    )
    restored, _ = load_checkpoint(path, "cpu")
    actual = restored.eval()(ids, routing_mode="greedy")
    torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
    torch.testing.assert_close(actual.mor_actions, expected.mor_actions)
    torch.testing.assert_close(actual.actions, expected.actions)
