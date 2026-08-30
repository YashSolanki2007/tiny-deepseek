"""Small decoder-only Transformers with dense or learned dynamic depth."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from router import GRUDepthRouter, MLPRouter, straight_through_gate


@dataclass
class ModelOutput:
    logits: torch.Tensor
    lm_loss: Optional[torch.Tensor] = None
    soft_gates: Optional[torch.Tensor] = None  # [B, T, L]
    hard_gates: Optional[torch.Tensor] = None  # [B, T, L]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Explicit masking keeps the implementation portable across PyTorch backends.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:, :, :seq_len, :seq_len], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.attn_dropout, training=self.training)
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.resid_dropout(self.output(attended))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    """A complete pre-norm residual block returning its candidate state."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.ln1(x))
        h = h + self.mlp(self.ln2(h))
        return h


def apply_block_gate(x: torch.Tensor, candidate: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Interpolate identity/candidate; an STE gate is binary in the forward pass."""
    # This algebraic form makes both binary endpoints bit-exact in floating point:
    # gate=0 returns x and gate=1 returns candidate, while d(output)/d(gate)
    # remains candidate-x for the straight-through estimator.
    return (1.0 - gate) * x + gate * candidate


class TransformerBase(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.shape[1] > self.config.context_length:
            raise ValueError("Sequence is longer than context_length")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        return self.embedding_dropout(
            self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        )

    def finish(
        self, x: torch.Tensor, targets: Optional[torch.Tensor], soft_gates=None, hard_gates=None
    ) -> ModelOutput:
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return ModelOutput(logits, loss, soft_gates, hard_gates)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class DenseTransformer(TransformerBase):
    def forward(self, token_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> ModelOutput:
        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.finish(x, targets)


class DynamicDepthTransformer(TransformerBase):
    """Stage A router: all candidates are evaluated, gates measure theoretical sparsity."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if config.router_type == "gru":
            self.router = GRUDepthRouter(
                config.d_model, config.router_dim, config.n_layers, config.gate_bias
            )
        else:
            self.router = MLPRouter(
                config.d_model, config.router_dim, config.n_layers, config.gate_bias
            )
        # The base initializer ran before the router existed, so initialize router defaults,
        # then restore the deliberately high gate biases and near-zero gate weights.
        self.router.apply(self._init_weights)
        heads = self.router.gate_heads if config.router_type == "gru" else [n[-1] for n in self.router.networks]
        for head in heads:
            nn.init.normal_(head.weight, mean=0.0, std=1e-3)
            nn.init.constant_(head.bias, config.gate_bias)

    def forward(self, token_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> ModelOutput:
        x = self.embed(token_ids)
        soft_gates = []
        hard_gates = []
        state = self.router.initial_state(x) if self.config.router_type == "gru" else None

        for layer_idx, block in enumerate(self.blocks):
            if self.config.router_type == "gru":
                logits, state = self.router.forward_layer(x, state, layer_idx)
            else:
                logits = self.router.forward_layer(x, layer_idx)
            gate, soft, hard = straight_through_gate(logits, self.config.gate_threshold)
            candidate = block(x)
            x = apply_block_gate(x, candidate, gate)
            soft_gates.append(soft.squeeze(-1))
            hard_gates.append(hard.squeeze(-1))

        return self.finish(
            x,
            targets,
            torch.stack(soft_gates, dim=-1),
            torch.stack(hard_gates, dim=-1),
        )


def build_model(config: ModelConfig) -> TransformerBase:
    if config.model_type == "dense":
        return DenseTransformer(config)
    return DynamicDepthTransformer(config)
