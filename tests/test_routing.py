from __future__ import annotations

import torch

from model import build_model
from tests.test_model import tiny_config


def test_linear_and_gru_routers_receive_supervised_gradients() -> None:
    for router_type in ("linear", "gru"):
        torch.manual_seed(7)
        model = build_model(tiny_config(router_type=router_type))
        x, y = torch.randint(0, 19, (2, 8)), torch.randint(0, 19, (2, 8))
        model(x, y, routing_mode="gumbel").lm_loss.backward()
        gradients = [parameter.grad for parameter in model.router.parameters()]
        assert any(
            gradient is not None and torch.count_nonzero(gradient).item() > 0
            for gradient in gradients
        )


def test_gru_state_from_one_depth_changes_the_next_depth() -> None:
    model = build_model(tiny_config(router_type="gru")).eval()
    x = torch.randn(2, 8, model.config.d_model)
    zero = model.router.initial_state(x)
    _, state_after_layer_zero = model.router.forward_layer(x, zero, 0)
    logits_with_history, _ = model.router.forward_layer(x, state_after_layer_zero, 1)
    logits_without_history, _ = model.router.forward_layer(x, zero, 1)
    assert not torch.allclose(state_after_layer_zero, zero)
    assert not torch.allclose(logits_with_history, logits_without_history)


def test_different_sampled_trajectories_can_differ() -> None:
    torch.manual_seed(13)
    model = build_model(tiny_config(router_type="gru")).eval()
    x = torch.randint(0, 19, (64, 8))
    actions = model(x, routing_mode="sample").actions
    assert torch.unique(actions.reshape(actions.shape[0], -1), dim=0).shape[0] > 1
