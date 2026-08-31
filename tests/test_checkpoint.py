from __future__ import annotations

import torch

from model import build_model
from tests.test_model import tiny_config
from utils import load_checkpoint, save_checkpoint


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
