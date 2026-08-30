"""Differentiable depth routers with hard straight-through gates."""

from __future__ import annotations

import torch
import torch.nn as nn


def straight_through_gate(
    logits: torch.Tensor, threshold: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    soft = torch.sigmoid(logits)
    hard = (soft >= threshold).to(dtype=soft.dtype)
    gate = soft + (hard - soft).detach()
    return gate, soft, hard


class GRUDepthRouter(nn.Module):
    """Maintains one GRU state per token and propagates it across depth."""

    def __init__(
        self, d_model: int, router_dim: int, n_layers: int, gate_bias: float = 2.2
    ) -> None:
        super().__init__()
        self.router_dim = router_dim
        self.input_projection = nn.Linear(d_model, router_dim)
        self.gru_cell = nn.GRUCell(router_dim, router_dim)
        self.gate_heads = nn.ModuleList([nn.Linear(router_dim, 1) for _ in range(n_layers)])
        for head in self.gate_heads:
            nn.init.normal_(head.weight, mean=0.0, std=1e-3)
            nn.init.constant_(head.bias, gate_bias)

    def initial_state(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            x.shape[0], x.shape[1], self.router_dim, device=x.device, dtype=x.dtype
        )

    def forward_layer(
        self, x: torch.Tensor, state: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        router_input = self.input_projection(x)
        state = self.gru_cell(
            router_input.reshape(batch * seq_len, self.router_dim),
            state.reshape(batch * seq_len, self.router_dim),
        ).reshape(batch, seq_len, self.router_dim)
        return self.gate_heads[layer_idx](state), state


class MLPRouter(nn.Module):
    """Independent per-layer router without explicit depth memory."""

    def __init__(
        self, d_model: int, router_dim: int, n_layers: int, gate_bias: float = 2.2
    ) -> None:
        super().__init__()
        self.networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, router_dim),
                    nn.GELU(),
                    nn.Linear(router_dim, 1),
                )
                for _ in range(n_layers)
            ]
        )
        for network in self.networks:
            output = network[-1]
            nn.init.normal_(output.weight, mean=0.0, std=1e-3)
            nn.init.constant_(output.bias, gate_bias)

    def forward_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.networks[layer_idx](x)
