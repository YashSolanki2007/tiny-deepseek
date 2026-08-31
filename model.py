"""Decoder-only dense and token-wise sparse-depth Transformers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from router import GRUDepthRouter, LinearDepthRouter, initialize_gate_head, route_from_logits


@dataclass
class ModelOutput:
    logits: torch.Tensor
    lm_loss: Optional[torch.Tensor] = None
    token_losses: Optional[torch.Tensor] = None
    soft_gates: Optional[torch.Tensor] = None
    hard_gates: Optional[torch.Tensor] = None
    actions: Optional[torch.Tensor] = None
    action_log_probs: Optional[torch.Tensor] = None
    behavior_log_probs: Optional[torch.Tensor] = None
    routing_entropy: Optional[torch.Tensor] = None
    route_logits: Optional[torch.Tensor] = None


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
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:, :, :seq_len, :seq_len], float("-inf"))
        weights = F.dropout(F.softmax(scores, dim=-1), p=self.attn_dropout, training=self.training)
        attended = (weights @ v).transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.resid_dropout(self.output(attended))

    def _projection(self, x: torch.Tensor, index: int) -> torch.Tensor:
        channels = self.n_heads * self.head_dim
        start, end = index * channels, (index + 1) * channels
        bias = self.qkv.bias[start:end] if self.qkv.bias is not None else None
        return F.linear(x, self.qkv.weight[start:end], bias)

    def forward_selected(
        self, normalized_x: torch.Tensor, active: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute K/V for all tokens and attention only for active queries."""
        batch, seq_len, channels = normalized_x.shape
        k = self._projection(normalized_x, 1).view(
            batch, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)
        v = self._projection(normalized_x, 2).view(
            batch, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)
        selected_outputs = []
        selected_indices = []
        key_positions = torch.arange(seq_len, device=normalized_x.device)
        for batch_index in range(batch):
            indices = active[batch_index].nonzero(as_tuple=False).flatten()
            selected_indices.append(indices)
            if indices.numel() == 0:
                selected_outputs.append(normalized_x.new_empty((0, channels)))
                continue
            selected = normalized_x[batch_index, indices]
            q = self._projection(selected, 0).view(
                indices.numel(), self.n_heads, self.head_dim
            ).transpose(0, 1)
            scores = (q @ k[batch_index].transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal = key_positions.unsqueeze(0) <= indices.unsqueeze(1)
            scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
            weights = F.softmax(scores, dim=-1)
            attended = (weights @ v[batch_index]).transpose(0, 1).contiguous().view(
                indices.numel(), channels
            )
            selected_outputs.append(self.resid_dropout(self.output(attended)))
        return selected_outputs, selected_indices


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
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.ln1(x))
        return h + self.mlp(self.ln2(h))

    def forward_selected(self, x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """Paper-faithful sparse inference for a hard `[B, T]` mask."""
        selected_attention, selected_indices = self.attn.forward_selected(self.ln1(x), active)
        output = x.clone()
        for batch_index, (attention, indices) in enumerate(
            zip(selected_attention, selected_indices)
        ):
            if indices.numel() == 0:
                continue
            h = x[batch_index, indices] + attention
            h = h + self.mlp(self.ln2(h))
            output[batch_index, indices] = h
        return output


def apply_block_gate(x: torch.Tensor, candidate: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
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
            self.token_embedding(token_ids) + self.position_embedding(positions)[None]
        )

    def finish(self, x: torch.Tensor, targets: Optional[torch.Tensor], **routing) -> ModelOutput:
        logits = self.lm_head(self.final_norm(x))
        token_losses = None
        loss = None
        if targets is not None:
            token_losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
            loss = token_losses.mean()
        return ModelOutput(logits=logits, lm_loss=loss, token_losses=token_losses, **routing)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def router_parameters(self):
        return iter(())


class DenseTransformer(TransformerBase):
    def forward(self, token_ids: torch.Tensor, targets: Optional[torch.Tensor] = None, **_) -> ModelOutput:
        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.finish(x, targets)


class SparseDepthTransformer(TransformerBase):
    """Stage A: gates are logically sparse, but every block candidate is evaluated."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if config.router_type == "gru":
            self.router = GRUDepthRouter(
                config.d_model,
                config.router_dim,
                config.n_layers,
                config.initial_execute_probability,
            )
        else:
            self.router = LinearDepthRouter(
                config.d_model,
                config.n_layers,
                config.initial_execute_probability,
                bias=config.router_bias,
            )
        self.router.apply(self._init_weights)
        for head in self.router.gate_heads:
            initialize_gate_head(
                head,
                config.initial_execute_probability,
                weight_std=0.02 if config.paper_reproduction else 1e-3,
            )

    def router_parameters(self):
        return self.router.parameters()

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        routing_mode: Optional[str] = None,
        actions: Optional[torch.Tensor] = None,
        router_override: Optional[nn.Module] = None,
        target_depths: Optional[torch.Tensor] = None,
        exploration_epsilon: float | torch.Tensor = 0.0,
    ) -> ModelOutput:
        x = self.embed(token_ids)
        router = router_override if router_override is not None else self.router
        mode = routing_mode or ("gumbel" if self.training else "greedy")
        state = router.initial_state(x) if self.config.router_type == "gru" else None
        soft_gates, hard_gates, chosen_actions = [], [], []
        log_probs, behavior_log_probs, entropies, logits_by_layer = [], [], [], []
        used_depth = torch.zeros(
            token_ids.shape, device=token_ids.device, dtype=torch.float32
        )
        if mode == "budget":
            if target_depths is None or target_depths.ndim != 1:
                raise ValueError("budget routing requires target_depths with shape [batch]")
            if target_depths.shape[0] != token_ids.shape[0]:
                raise ValueError("target_depths batch dimension must match token_ids")
            target_depths = target_depths.to(device=token_ids.device, dtype=torch.float32)
            if bool(((target_depths < 0) | (target_depths > self.config.n_layers)).any()):
                raise ValueError("target_depths must lie between 0 and n_layers")
            epsilon = torch.as_tensor(
                exploration_epsilon, device=token_ids.device, dtype=torch.float32
            )
            if epsilon.ndim == 0:
                epsilon = epsilon.expand(token_ids.shape[0])
            if epsilon.shape != (token_ids.shape[0],):
                raise ValueError("exploration_epsilon must be scalar or shape [batch]")
            if bool(((epsilon < 0) | (epsilon > 1)).any()):
                raise ValueError("exploration_epsilon must lie in [0, 1]")
        for layer_idx, block in enumerate(self.blocks):
            router_input = block.ln1(x) if self.config.router_input_norm else x
            if self.config.router_type == "gru":
                logits, state = router.forward_layer(router_input, state, layer_idx)
            else:
                logits = router.forward_layer(router_input, layer_idx)
            supplied = actions[..., layer_idx] if actions is not None else None
            if mode == "budget" and supplied is None:
                scaled = logits / self.config.gumbel_temperature
                policy_log_probs = F.log_softmax(scaled, dim=-1)
                policy_probs = policy_log_probs.exp()
                remaining = float(self.config.n_layers - layer_idx)
                needed = (target_depths[:, None] - used_depth).clamp(0.0, remaining)
                controller_execute = needed / remaining
                mixed_execute = (
                    (1.0 - epsilon[:, None]) * policy_probs[..., 1]
                    + epsilon[:, None] * controller_execute
                )
                behavior_execute = mixed_execute.clamp(
                    torch.finfo(logits.dtype).eps,
                    1.0 - torch.finfo(logits.dtype).eps,
                )
                behavior_execute = torch.where(
                    epsilon[:, None].eq(1.0), controller_execute, behavior_execute
                )
                chosen = torch.rand_like(behavior_execute).lt(behavior_execute).long()
                gate = chosen.to(logits.dtype)
                soft = policy_probs[..., 1]
                log_prob = torch.where(
                    chosen.bool(), policy_log_probs[..., 1], policy_log_probs[..., 0]
                )
                behavior_log_prob = torch.where(
                    chosen.bool(), behavior_execute.log(), (-behavior_execute).log1p()
                )
                entropy = -(policy_probs * policy_log_probs).sum(dim=-1)
                used_depth = used_depth + chosen.float()
            else:
                gate, soft, chosen, log_prob, entropy = route_from_logits(
                    logits, mode, self.config.gumbel_temperature, supplied
                )
                behavior_log_prob = log_prob
            if (
                self.config.sparse_inference
                and not self.training
                and mode == "greedy"
            ):
                x = block.forward_selected(x, chosen.bool())
            else:
                candidate = block(x)
                x = apply_block_gate(x, candidate, gate.unsqueeze(-1))
            soft_gates.append(soft)
            hard_gates.append(gate)
            chosen_actions.append(chosen)
            log_probs.append(log_prob)
            behavior_log_probs.append(behavior_log_prob)
            entropies.append(entropy)
            logits_by_layer.append(logits)
        return self.finish(
            x,
            targets,
            soft_gates=torch.stack(soft_gates, dim=-1),
            hard_gates=torch.stack(hard_gates, dim=-1),
            actions=torch.stack(chosen_actions, dim=-1),
            action_log_probs=torch.stack(log_probs, dim=-1),
            behavior_log_probs=torch.stack(behavior_log_probs, dim=-1),
            routing_entropy=torch.stack(entropies, dim=-1),
            route_logits=torch.stack(logits_by_layer, dim=2),
        )


DynamicDepthTransformer = SparseDepthTransformer


def build_model(config: ModelConfig) -> TransformerBase:
    return DenseTransformer(config) if config.model_type == "dense" else SparseDepthTransformer(config)
