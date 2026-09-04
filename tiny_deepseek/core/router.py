"""Linear and depth-recurrent binary routing policies."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_gumbel_action(logits: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a binary action and ST one-hot without MPS scatter kernels."""
    epsilon = torch.finfo(logits.dtype).eps
    uniform = torch.rand_like(logits).clamp(min=epsilon, max=1.0 - epsilon)
    gumbel = -torch.log(-torch.log(uniform))
    relaxed = F.softmax((logits + gumbel) / temperature, dim=-1)
    execute = relaxed[..., 1].ge(relaxed[..., 0])
    hard = torch.stack((~execute, execute), dim=-1).to(logits.dtype)
    straight_through = relaxed + (hard - relaxed).detach()
    return execute.long(), straight_through


def initialize_gate_head(
    head: nn.Linear, execute_probability: float, weight_std: float = 1e-3
) -> None:
    nn.init.normal_(head.weight, mean=0.0, std=weight_std)
    if head.bias is not None:
        nn.init.zeros_(head.bias)
        with torch.no_grad():
            head.bias[1] = math.log(execute_probability / (1 - execute_probability))


def route_from_logits(
    logits: torch.Tensor,
    mode: str,
    temperature: float = 1.0,
    action: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return execute gate/probability, action, selected log-p, and entropy."""
    scaled = logits / temperature
    log_probabilities = F.log_softmax(scaled, dim=-1)
    probabilities = log_probabilities.exp()
    if action is not None:
        chosen = action.long()
        gate = chosen.to(logits.dtype)
    elif mode == "gumbel":
        chosen, onehot = binary_gumbel_action(logits, temperature)
        gate = onehot[..., 1]
    elif mode == "sample":
        # The temperature-scaled Gumbel-max action is a categorical sample.
        chosen, _ = binary_gumbel_action(scaled, 1.0)
        gate = chosen.to(logits.dtype)
    elif mode == "greedy":
        chosen = logits[..., 1].ge(logits[..., 0]).long()
        gate = chosen.to(logits.dtype)
    else:
        raise ValueError(f"Unknown routing mode: {mode}")
    selected_log_probability = torch.where(
        chosen.bool(), log_probabilities[..., 1], log_probabilities[..., 0]
    )
    entropy = -(probabilities * log_probabilities).sum(dim=-1)
    return gate, probabilities[..., 1], chosen, selected_log_probability, entropy


class LinearDepthRouter(nn.Module):
    """SkipLayer-style independent linear router at every depth."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        execute_probability: float = 0.9,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.gate_heads = nn.ModuleList(
            [nn.Linear(d_model, 2, bias=bias) for _ in range(n_layers)]
        )
        for head in self.gate_heads:
            initialize_gate_head(head, execute_probability)

    def forward_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.gate_heads[layer_idx](x)


class GRUDepthRouter(nn.Module):
    """Maintains one hidden state per token and propagates it across model depth."""

    def __init__(
        self,
        d_model: int,
        router_dim: int,
        n_layers: int,
        execute_probability: float = 0.9,
    ) -> None:
        super().__init__()
        self.router_dim = router_dim
        self.input_projection = nn.Linear(d_model, router_dim)
        self.gru_cell = nn.GRUCell(router_dim, router_dim)
        self.gate_heads = nn.ModuleList([nn.Linear(router_dim, 2) for _ in range(n_layers)])
        for head in self.gate_heads:
            initialize_gate_head(head, execute_probability)

    def initial_state(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            x.shape[0], x.shape[1], self.router_dim, device=x.device, dtype=x.dtype
        )

    def forward_layer(
        self, x: torch.Tensor, state: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        projected = self.input_projection(x)
        state = self.gru_cell(
            projected.reshape(batch * seq_len, self.router_dim),
            state.reshape(batch * seq_len, self.router_dim),
        ).reshape(batch, seq_len, self.router_dim)
        return self.gate_heads[layer_idx](state), state
