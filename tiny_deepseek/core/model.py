"""Decoder-only dense and token-wise sparse-depth Transformers."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tiny_deepseek.core.config import ModelConfig
from tiny_deepseek.core.router import GRUDepthRouter, LinearDepthRouter, initialize_gate_head, route_from_logits


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
    routing_decision_mask: Optional[torch.Tensor] = None
    mor_aux_loss: Optional[torch.Tensor] = None
    mor_router_accuracy: Optional[torch.Tensor] = None
    recursion_utilization: Optional[torch.Tensor] = None
    recursion_soft_utilization: Optional[torch.Tensor] = None
    mor_actions: Optional[torch.Tensor] = None
    skip_hard_gates: Optional[torch.Tensor] = None
    skip_soft_gates: Optional[torch.Tensor] = None
    skip_conditional_utilization: Optional[torch.Tensor] = None
    skip_soft_conditional_utilization: Optional[torch.Tensor] = None
    combined_block_utilization: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    moe_aux_loss: Optional[torch.Tensor] = None
    moe_router_entropy: Optional[torch.Tensor] = None
    expert_utilization: Optional[torch.Tensor] = None
    expert_affinity: Optional[torch.Tensor] = None
    mtp_logits: Optional[torch.Tensor] = None
    mtp_loss: Optional[torch.Tensor] = None
    mtp_accuracy: Optional[torch.Tensor] = None


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


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.epsilon
        )
        return normalized.to(x.dtype) * self.weight


def apply_rotary_embedding(
    x: torch.Tensor, theta: float, position_offset: int = 0
) -> torch.Tensor:
    """Apply interleaved RoPE to `[batch, heads, sequence, dimension]`."""
    dimension = x.shape[-1]
    if dimension % 2:
        raise ValueError("RoPE dimension must be even")
    positions = torch.arange(
        position_offset,
        position_offset + x.shape[-2],
        device=x.device,
        dtype=torch.float32,
    )
    inverse_frequency = 1.0 / (
        theta ** (torch.arange(0, dimension, 2, device=x.device).float() / dimension)
    )
    angles = torch.outer(positions, inverse_frequency)
    cosine = angles.cos()[None, None]
    sine = angles.sin()[None, None]
    even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
    rotated = torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    ).flatten(-2)
    return rotated.to(x.dtype)


class MultiHeadLatentAttention(nn.Module):
    """DeepSeek-style MLA with partial RoPE and PyTorch fused SDPA.

    KV content is compressed into one shared latent vector per token. Positional
    key dimensions bypass that bottleneck and are shared across heads, matching
    MLA's decoupled-RoPE construction.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.q_rank = config.mla_q_lora_rank
        self.kv_rank = config.mla_kv_lora_rank
        self.nope_dim = config.mla_qk_nope_head_dim
        self.rope_dim = config.mla_qk_rope_head_dim
        self.qk_dim = self.nope_dim + self.rope_dim
        self.value_dim = config.mla_v_head_dim
        self.theta = config.rope_theta
        self.dropout = config.dropout
        if self.q_rank > 0:
            self.q_down = nn.Linear(config.d_model, self.q_rank, bias=False)
            self.q_norm = RMSNorm(self.q_rank)
            self.q_up = nn.Linear(self.q_rank, self.n_heads * self.qk_dim, bias=False)
        else:
            self.q_proj = nn.Linear(config.d_model, self.n_heads * self.qk_dim, bias=False)
        self.kv_down = nn.Linear(
            config.d_model, self.kv_rank + self.rope_dim, bias=False
        )
        self.kv_norm = RMSNorm(self.kv_rank)
        self.kv_up = nn.Linear(
            self.kv_rank,
            self.n_heads * (self.nope_dim + self.value_dim),
            bias=False,
        )
        self.output = nn.Linear(self.n_heads * self.value_dim, config.d_model, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _project(
        self, x: torch.Tensor, position_offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        if self.q_rank > 0:
            q = self.q_up(self.q_norm(self.q_down(x)))
        else:
            q = self.q_proj(x)
        q = q.view(batch, seq_len, self.n_heads, self.qk_dim).transpose(1, 2)
        q_nope, q_rope = q.split((self.nope_dim, self.rope_dim), dim=-1)
        q_rope = apply_rotary_embedding(q_rope, self.theta, position_offset)

        latent_and_rope = self.kv_down(x)
        latent, k_rope = latent_and_rope.split((self.kv_rank, self.rope_dim), dim=-1)
        expanded = self.kv_up(self.kv_norm(latent))
        expanded = expanded.view(
            batch, seq_len, self.n_heads, self.nope_dim + self.value_dim
        ).transpose(1, 2)
        k_nope, value = expanded.split((self.nope_dim, self.value_dim), dim=-1)
        k_rope = apply_rotary_embedding(k_rope[:, None], self.theta, position_offset)
        key = torch.cat((k_nope, k_rope.expand(-1, self.n_heads, -1, -1)), dim=-1)
        query = torch.cat((q_nope, q_rope), dim=-1)
        return query, key, value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query, key, value = self._project(x)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().flatten(2)
        return self.resid_dropout(self.output(attended))

    def forward_cached(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """MLA forward with an inference KV cache for full-depth decoding."""
        position_offset = 0 if cache is None else cache[0].shape[-2]
        query, new_key, new_value = self._project(x, position_offset)
        if cache is None:
            key, value = new_key, new_value
        else:
            key = torch.cat((cache[0], new_key), dim=-2)
            value = torch.cat((cache[1], new_value), dim=-2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=cache is None and x.shape[1] > 1,
        )
        attended = attended.transpose(1, 2).contiguous().flatten(2)
        return self.resid_dropout(self.output(attended)), (key, value)

    def forward_selected(
        self, normalized_x: torch.Tensor, active: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Keep every token as KV context while computing only active queries."""
        query, key, value = self._project(normalized_x)
        seq_len = normalized_x.shape[1]
        key_positions = torch.arange(seq_len, device=normalized_x.device)
        selected_outputs, selected_indices = [], []
        for batch_index in range(normalized_x.shape[0]):
            indices = active[batch_index].nonzero(as_tuple=False).flatten()
            selected_indices.append(indices)
            if indices.numel() == 0:
                selected_outputs.append(
                    normalized_x.new_empty((0, self.n_heads * self.value_dim))
                )
                continue
            mask = key_positions[None, :] <= indices[:, None]
            attended = F.scaled_dot_product_attention(
                query[batch_index : batch_index + 1, :, indices],
                key[batch_index : batch_index + 1],
                value[batch_index : batch_index + 1],
                attn_mask=mask[None, None],
                dropout_p=0.0,
                is_causal=False,
            )
            attended = attended.transpose(1, 2).contiguous().flatten(2)[0]
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


class SparseMoE(nn.Module):
    """Fine-grained top-k FFN experts with DeepSeek-style routing biases."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        expert_config = replace(config, d_ff=config.moe_expert_d_ff, model_type="dense")
        self.experts = nn.ModuleList([MLP(expert_config) for _ in range(self.num_experts)])
        self.router = nn.Linear(config.d_model, self.num_experts, bias=False)
        self.register_buffer("selection_bias", torch.zeros(self.num_experts))
        self.last_aux_loss: Optional[torch.Tensor] = None
        self.last_entropy: Optional[torch.Tensor] = None
        self.last_utilization: Optional[torch.Tensor] = None
        self.last_affinity: Optional[torch.Tensor] = None

    def clear_stats(self, reference: torch.Tensor) -> None:
        zero = reference.sum() * 0.0
        self.last_aux_loss = zero
        self.last_entropy = zero
        self.last_utilization = reference.new_zeros(self.num_experts)
        self.last_affinity = reference.new_zeros(self.num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        if flat.shape[0] == 0:
            self.clear_stats(x)
            return x
        affinity = torch.sigmoid(self.router(flat))
        selected = torch.topk(
            affinity + self.selection_bias.to(affinity.dtype), self.top_k, dim=-1
        ).indices
        selected_affinity = affinity.gather(-1, selected)
        weights = selected_affinity / selected_affinity.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(selected_affinity.dtype).eps
        )
        output = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            token_index, slot_index = (selected == expert_index).nonzero(as_tuple=True)
            if token_index.numel() == 0:
                continue
            contribution = expert(flat[token_index]) * weights[token_index, slot_index, None]
            output.index_add_(0, token_index, contribution)

        assignments = F.one_hot(selected, self.num_experts).float().sum(dim=1)
        utilization = assignments.mean(dim=0) / float(self.top_k)
        normalized_affinity = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(affinity.dtype).eps
        )
        mean_affinity = normalized_affinity.mean(dim=0)
        frequency = assignments.mean(dim=0) * (self.num_experts / float(self.top_k))
        self.last_aux_loss = (frequency * mean_affinity).sum()
        self.last_entropy = -(
            normalized_affinity * normalized_affinity.clamp_min(1e-9).log()
        ).sum(dim=-1).mean()
        self.last_utilization = utilization.detach()
        self.last_affinity = mean_affinity.detach()
        return output.reshape(original_shape)

    @torch.no_grad()
    def update_selection_bias(self, speed: float) -> None:
        if speed <= 0 or self.last_utilization is None:
            return
        target = self.last_utilization.new_full(
            self.last_utilization.shape, 1.0 / self.num_experts
        )
        self.selection_bias.add_(speed * torch.sign(target - self.last_utilization))
        self.selection_bias.sub_(self.selection_bias.mean())


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = (
            MultiHeadLatentAttention(config)
            if config.attention_type == "mla"
            else CausalSelfAttention(config)
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = (
            SparseMoE(config)
            if config.model_type in {"sparse_moe_mtp", "sparse_moe_mtp_mla"}
            else MLP(config)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.ln1(x))
        return h + self.mlp(self.ln2(h))

    def forward_cached(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if not isinstance(self.attn, MultiHeadLatentAttention):
            raise TypeError("cached decoding is currently implemented for MLA only")
        attention, updated_cache = self.attn.forward_cached(self.ln1(x), cache)
        h = x + attention
        return h + self.mlp(self.ln2(h)), updated_cache

    def forward_selected(self, x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """Paper-faithful sparse inference for a hard `[B, T]` mask."""
        selected_attention, selected_indices = self.attn.forward_selected(self.ln1(x), active)
        output = x.clone()
        hidden = [
            x[batch_index, indices] + attention
            for batch_index, (attention, indices) in enumerate(
                zip(selected_attention, selected_indices)
            )
            if indices.numel() > 0
        ]
        if not hidden:
            if isinstance(self.mlp, SparseMoE):
                self.mlp.clear_stats(x)
            return output
        transformed = torch.cat(hidden, dim=0)
        transformed = transformed + self.mlp(self.ln2(transformed))
        offset = 0
        for batch_index, indices in enumerate(selected_indices):
            count = indices.numel()
            if count:
                output[batch_index, indices] = transformed[offset : offset + count]
                offset += count
        return output


def apply_block_gate(x: torch.Tensor, candidate: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (1.0 - gate) * x + gate * candidate


class TransformerBase(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = (
            nn.Embedding(config.context_length, config.d_model)
            if config.position_embedding_type == "learned" else None
        )
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
        embedded = self.token_embedding(token_ids)
        if self.position_embedding is not None:
            positions = torch.arange(token_ids.shape[1], device=token_ids.device)
            embedded = embedded + self.position_embedding(positions)[None]
        return self.embedding_dropout(embedded)

    def finish(self, x: torch.Tensor, targets: Optional[torch.Tensor], **routing) -> ModelOutput:
        logits = self.lm_head(self.final_norm(x))
        token_losses = None
        loss = None
        if targets is not None:
            token_losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
            valid_targets = targets.ne(-100)
            loss = token_losses.sum() / valid_targets.sum().clamp_min(1)
        return ModelOutput(
            logits=logits, lm_loss=loss, token_losses=token_losses,
            hidden_states=x, **routing
        )

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
        moe_aux_losses, moe_entropies = [], []
        expert_utilization, expert_affinity = [], []
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
            if self.config.sparse_inference and not self.training and mode == "greedy":
                x = block.forward_selected(x, chosen.bool())
            else:
                candidate = block(x)
                x = apply_block_gate(x, candidate, gate.unsqueeze(-1))
            if isinstance(block.mlp, SparseMoE):
                moe_aux_losses.append(block.mlp.last_aux_loss)
                moe_entropies.append(block.mlp.last_entropy)
                expert_utilization.append(block.mlp.last_utilization)
                expert_affinity.append(block.mlp.last_affinity)
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
            moe_aux_loss=(
                torch.stack(moe_aux_losses).mean() if moe_aux_losses else None
            ),
            moe_router_entropy=(
                torch.stack(moe_entropies).mean() if moe_entropies else None
            ),
            expert_utilization=(
                torch.stack(expert_utilization) if expert_utilization else None
            ),
            expert_affinity=(
                torch.stack(expert_affinity) if expert_affinity else None
            ),
        )


class SparseMoEMTPTransformer(SparseDepthTransformer):
    """SkipLayer with dense or sparse FFNs, optional MLA+RoPE, and one MTP depth."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        dense_config = replace(config, model_type="dense")
        self.mtp_hidden_norm = nn.LayerNorm(config.d_model)
        self.mtp_token_norm = nn.LayerNorm(config.d_model)
        self.mtp_projection = nn.Linear(2 * config.d_model, config.d_model, bias=False)
        self.mtp_block = TransformerBlock(dense_config)
        self.mtp_output_norm = nn.LayerNorm(config.d_model)
        self.mtp_projection.apply(self._init_weights)
        self.mtp_block.apply(self._init_weights)

    def update_moe_selection_biases(self) -> None:
        for block in self.blocks:
            if isinstance(block.mlp, SparseMoE):
                block.mlp.update_selection_bias(self.config.moe_bias_update_speed)

    def full_depth_prefill(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Prefill all MLA layers and return next-token logits plus KV caches."""
        if self.config.attention_type != "mla":
            raise TypeError("full-depth cached decoding requires MLA")
        x = self.embed(token_ids)
        caches = []
        for block in self.blocks:
            x, cache = block.forward_cached(x)
            caches.append(cache)
        logits = self.lm_head(self.final_norm(x[:, -1]))
        return logits, caches

    def full_depth_decode(
        self,
        next_token: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Decode one token through all layers and extend the MLA KV caches."""
        if next_token.ndim == 1:
            next_token = next_token[:, None]
        if next_token.shape[1] != 1 or len(caches) != len(self.blocks):
            raise ValueError("cached decode expects one token and one cache per layer")
        x = self.embed(next_token)
        updated = []
        for block, cache in zip(self.blocks, caches):
            x, new_cache = block.forward_cached(x, cache)
            updated.append(new_cache)
        logits = self.lm_head(self.final_norm(x[:, -1]))
        return logits, updated

    def mtp_draft_logits(
        self,
        token_ids: torch.Tensor,
        main_hidden_states: torch.Tensor,
        proposed_next_token: torch.Tensor,
    ) -> torch.Tensor:
        """Draft the token after ``proposed_next_token`` with the one-block MTP head.

        ``main_hidden_states`` must be the full-model states for ``token_ids``.
        The shifted observed tokens preserve the MTP module's complete causal chain.
        """
        if token_ids.ndim != 2 or main_hidden_states.shape[:2] != token_ids.shape:
            raise ValueError("token_ids and main_hidden_states must share [batch, sequence]")
        if proposed_next_token.shape not in {
            (token_ids.shape[0],), (token_ids.shape[0], 1)
        }:
            raise ValueError("proposed_next_token must have shape [batch] or [batch, 1]")
        proposed_next_token = proposed_next_token.reshape(token_ids.shape[0], 1)
        following_tokens = torch.cat((token_ids[:, 1:], proposed_next_token), dim=1)
        main_hidden = self.mtp_hidden_norm(main_hidden_states)
        next_token = self.mtp_token_norm(self.token_embedding(following_tokens))
        fused = self.mtp_projection(torch.cat((main_hidden, next_token), dim=-1))
        mtp_hidden = self.mtp_block(fused)
        return self.lm_head(self.mtp_output_norm(mtp_hidden))[:, -1]

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        compute_mtp: bool = True,
        **routing,
    ) -> ModelOutput:
        output = super().forward(token_ids, targets, **routing)
        if targets is None or not compute_mtp or targets.shape[1] < 2:
            return output
        main_hidden = self.mtp_hidden_norm(output.hidden_states[:, :-1])
        # Use the unmasked observed next tokens. Supervised targets may contain
        # ignore_index=-100 over prompts and padding, which cannot be embedded.
        next_token = self.mtp_token_norm(self.token_embedding(token_ids[:, 1:]))
        fused = self.mtp_projection(torch.cat((main_hidden, next_token), dim=-1))
        mtp_hidden = self.mtp_block(fused)
        mtp_logits = self.lm_head(self.mtp_output_norm(mtp_hidden))
        mtp_targets = targets[:, 1:]
        output.mtp_logits = mtp_logits
        output.mtp_loss = F.cross_entropy(
            mtp_logits.transpose(1, 2), mtp_targets
        )
        valid_mtp = mtp_targets.ne(-100)
        output.mtp_accuracy = (
            mtp_logits.argmax(dim=-1).eq(mtp_targets)[valid_mtp].float().mean()
            if bool(valid_mtp.any())
            else mtp_logits.sum() * 0.0
        )
        return output


class MixtureOfRecursionsTransformer(TransformerBase):
    """Tiny-Shakespeare analogue of expert-choice Middle-Cycle MoR.

    The effective stack is ``entry + shared_block * recursions + exit``.
    Training uses hierarchical expert-choice top-k routing and the paper's
    auxiliary BCE. Greedy evaluation uses the learned 0.5 threshold, avoiding
    validation-time top-k information leakage.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        initialized = list(self.blocks)
        unique_count = config.recursion_block_layers + 2
        self.blocks = nn.ModuleList(initialized[:unique_count])
        self.recursion_routers = nn.ModuleList(
            [nn.Linear(config.d_model, 1) for _ in range(config.recursion_steps)]
        )
        self.recursion_routers.apply(self._init_weights)

    @property
    def router(self) -> nn.ModuleList:
        return self.recursion_routers

    def router_parameters(self):
        return self.recursion_routers.parameters()

    def _capacity(self, recursion_index: int) -> float:
        target = self.config.mor_capacity_factors[recursion_index]
        if not self.training or self.config.mor_capacity_warmup_steps <= 0:
            return target
        # Kept for compatibility with the authors' cosine capacity warmup.
        # The selected paper configuration uses zero warmup steps.
        return target

    @staticmethod
    def _selected_log_probability(
        binary_logits: torch.Tensor, selected: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probabilities = F.log_softmax(binary_logits, dim=-1)
        probabilities = log_probabilities.exp()
        log_probability = torch.where(
            selected, log_probabilities[..., 1], log_probabilities[..., 0]
        )
        entropy = -(probabilities * log_probabilities).sum(dim=-1)
        return probabilities[..., 1], log_probability, entropy

    def _apply_recursion_block(
        self, x: torch.Tensor, selected: torch.Tensor, gate_weight: torch.Tensor
    ) -> torch.Tensor:
        output = x.clone()
        shared_blocks = self.blocks[1:-1]
        for batch_index in range(x.shape[0]):
            indices = selected[batch_index].nonzero(as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            initial = x[batch_index : batch_index + 1, indices]
            transformed = initial
            for block in shared_blocks:
                transformed = block(transformed)
            weight = gate_weight[batch_index, indices].view(1, -1, 1)
            updated = initial + weight * (transformed - initial)
            output[batch_index, indices] = updated[0]
        return output

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        routing_mode: Optional[str] = None,
        actions: Optional[torch.Tensor] = None,
        router_override: Optional[nn.Module] = None,
        target_recursions: Optional[torch.Tensor] = None,
        exploration_epsilon: float | torch.Tensor = 0.0,
    ) -> ModelOutput:
        mode = routing_mode or ("topk" if self.training else "greedy")
        if mode == "gumbel":
            mode = "topk"
        routers = router_override if router_override is not None else self.recursion_routers
        x = self.blocks[0](self.embed(token_ids))
        batch, seq_len = token_ids.shape
        previous_selected = torch.ones(
            batch, seq_len, device=token_ids.device, dtype=torch.bool
        )
        soft_by_recursion, hard_by_recursion = [], []
        actions_by_recursion, log_probs, behavior_log_probs = [], [], []
        entropies, logits_by_recursion, decision_masks = [], [], []
        auxiliary_losses, accuracies = [], []

        if mode == "budget":
            if target_recursions is None or target_recursions.shape != (batch,):
                raise ValueError("budget MoR routing requires target_recursions shaped [batch]")
            target_recursions = target_recursions.to(token_ids.device)
            if bool(
                ((target_recursions < 1) | (target_recursions > self.config.recursion_steps)).any()
            ):
                raise ValueError("target_recursions must lie in [1, recursion_steps]")
            epsilon = torch.as_tensor(
                exploration_epsilon, device=token_ids.device, dtype=torch.float32
            )
            if epsilon.ndim == 0:
                epsilon = epsilon.expand(batch)
            if epsilon.shape != (batch,) or bool(((epsilon < 0) | (epsilon > 1)).any()):
                raise ValueError("exploration_epsilon must be scalar or a [batch] value in [0,1]")

        for recursion_index in range(self.config.recursion_steps):
            raw_logits = routers[recursion_index](x).squeeze(-1)
            binary_logits = torch.stack((torch.zeros_like(raw_logits), raw_logits), dim=-1)
            policy_execute = raw_logits.sigmoid()
            decision_mask = previous_selected.clone()
            supplied = actions[..., recursion_index].bool() if actions is not None else None

            if supplied is not None:
                selected = supplied & previous_selected
                behavior_log_probability = None
            elif recursion_index == 0:
                selected = previous_selected
                behavior_log_probability = torch.zeros_like(raw_logits)
                decision_mask = torch.zeros_like(previous_selected)
            elif mode == "topk":
                capacity = self._capacity(recursion_index)
                count = max(1, int(capacity * seq_len))
                masked_scores = raw_logits.masked_fill(~previous_selected, float("-inf"))
                selected = torch.zeros_like(previous_selected)
                for batch_index in range(batch):
                    available = int(previous_selected[batch_index].sum().item())
                    take = min(count, available)
                    if take:
                        indices = masked_scores[batch_index].topk(take, sorted=False).indices
                        selected[batch_index, indices] = True
                behavior_log_probability = None
            elif mode == "greedy":
                selected = policy_execute.ge(0.5) & previous_selected
                behavior_log_probability = None
            elif mode == "sample":
                selected = torch.rand_like(policy_execute).lt(policy_execute) & previous_selected
                behavior_log_probability = None
            elif mode == "budget":
                controller_execute = target_recursions[:, None].gt(recursion_index)
                mixed = (
                    (1.0 - epsilon[:, None]) * policy_execute
                    + epsilon[:, None] * controller_execute.float()
                )
                behavior_execute = mixed.clamp(
                    torch.finfo(raw_logits.dtype).eps,
                    1.0 - torch.finfo(raw_logits.dtype).eps,
                )
                behavior_execute = torch.where(
                    epsilon[:, None].eq(1.0),
                    controller_execute.float().expand_as(behavior_execute),
                    behavior_execute,
                )
                selected = torch.rand_like(behavior_execute).lt(behavior_execute)
                selected = selected & previous_selected
                behavior_log_probability = torch.where(
                    selected, behavior_execute.log(), (-behavior_execute).log1p()
                )
            else:
                raise ValueError(f"Unknown MoR routing mode: {mode}")

            soft, log_probability, entropy = self._selected_log_probability(
                binary_logits, selected
            )
            if behavior_log_probability is None:
                behavior_log_probability = log_probability

            capacity = self._capacity(recursion_index)
            target_count = max(1, int(capacity * seq_len))
            oracle_target = torch.zeros_like(previous_selected)
            masked_scores = raw_logits.masked_fill(~previous_selected, float("-inf"))
            for batch_index in range(batch):
                available = int(previous_selected[batch_index].sum().item())
                take = min(target_count, available)
                if take:
                    indices = masked_scores[batch_index].topk(take, sorted=False).indices
                    oracle_target[batch_index, indices] = True
            candidates = previous_selected
            if bool(candidates.any()):
                auxiliary_losses.append(
                    F.binary_cross_entropy_with_logits(
                        raw_logits[candidates], oracle_target[candidates].float()
                    )
                )
                accuracies.append(
                    policy_execute.ge(0.5)[candidates]
                    .eq(oracle_target[candidates])
                    .float()
                    .mean()
                )
            else:
                auxiliary_losses.append(raw_logits.sum() * 0.0)
                accuracies.append(raw_logits.new_ones(()))

            gate_weight = self.config.mor_router_alpha * policy_execute
            x = self._apply_recursion_block(x, selected, gate_weight)
            previous_selected = selected
            soft_by_recursion.append(soft * candidates.float())
            hard_by_recursion.append(selected.float())
            actions_by_recursion.append(selected.long())
            log_probs.append(log_probability)
            behavior_log_probs.append(behavior_log_probability)
            entropies.append(entropy)
            logits_by_recursion.append(binary_logits)
            decision_masks.append(decision_mask)

        x = self.blocks[-1](x)
        ones = torch.ones_like(hard_by_recursion[0])
        hard_layers = [ones]
        soft_layers = [ones]
        for soft, hard in zip(soft_by_recursion, hard_by_recursion):
            hard_layers.extend([hard] * self.config.recursion_block_layers)
            soft_layers.extend([soft] * self.config.recursion_block_layers)
        hard_layers.append(ones)
        soft_layers.append(ones)
        recursion_hard = torch.stack(hard_by_recursion, dim=-1)
        recursion_soft = torch.stack(soft_by_recursion, dim=-1)
        return self.finish(
            x,
            targets,
            soft_gates=torch.stack(soft_layers, dim=-1),
            hard_gates=torch.stack(hard_layers, dim=-1),
            actions=torch.stack(actions_by_recursion, dim=-1),
            action_log_probs=torch.stack(log_probs, dim=-1),
            behavior_log_probs=torch.stack(behavior_log_probs, dim=-1),
            routing_entropy=torch.stack(entropies, dim=-1),
            route_logits=torch.stack(logits_by_recursion, dim=2),
            routing_decision_mask=torch.stack(decision_masks, dim=-1),
            mor_aux_loss=torch.stack(auxiliary_losses).sum(),
            mor_router_accuracy=torch.stack(accuracies).mean(),
            recursion_utilization=recursion_hard.float().mean(dim=(0, 1)),
            recursion_soft_utilization=recursion_soft.float().mean(dim=(0, 1)),
        )


class MixtureOfRecursionsSkipLayerTransformer(MixtureOfRecursionsTransformer):
    """Paper-style MoR with an independent SkipLayer gate inside each cycle.

    MoR first admits tokens to a recursion. Each admitted token then receives a
    separate binary skip/execute decision for every shared block application.
    Entry and exit blocks always execute. The six inner heads in the default
    2-block/3-recursion setup are distinct even though the block parameters are
    shared across recursions.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        inner_layers = config.recursion_steps * config.recursion_block_layers
        self.skip_router = LinearDepthRouter(
            config.d_model,
            inner_layers,
            config.initial_execute_probability,
            bias=config.router_bias,
        )
        self.skip_router.apply(self._init_weights)
        for head in self.skip_router.gate_heads:
            initialize_gate_head(
                head,
                config.initial_execute_probability,
                weight_std=0.02 if config.paper_reproduction else 1e-3,
            )

    @property
    def router(self) -> LinearDepthRouter:
        """Expose the trainable inner SkipLayer policy as the primary router."""
        return self.skip_router

    def skip_router_parameters(self):
        return self.skip_router.parameters()

    def mor_router_parameters(self):
        return self.recursion_routers.parameters()

    def router_parameters(self):
        return iter((*self.recursion_routers.parameters(), *self.skip_router.parameters()))

    @staticmethod
    def _conditional_mean(
        values: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        numerator = (values * candidates.to(values.dtype)).sum(dim=(0, 1))
        denominator = candidates.sum(dim=(0, 1)).clamp_min(1).to(values.dtype)
        return numerator / denominator

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        routing_mode: Optional[str] = None,
        actions: Optional[torch.Tensor] = None,
        router_override: Optional[nn.Module] = None,
        target_skip_densities: Optional[torch.Tensor] = None,
        exploration_epsilon: float | torch.Tensor = 0.0,
    ) -> ModelOutput:
        mode = routing_mode or ("topk" if self.training else "greedy")
        if mode == "gumbel":
            mode = "topk"
        if mode not in {"topk", "greedy", "sample", "budget"}:
            raise ValueError(f"Unknown MoR+Skip routing mode: {mode}")

        skip_router = router_override if router_override is not None else self.skip_router
        x = self.blocks[0](self.embed(token_ids))
        batch, seq_len = token_ids.shape
        inner_count = self.config.recursion_steps * self.config.recursion_block_layers
        if actions is not None and actions.shape != (batch, seq_len, inner_count):
            raise ValueError(
                "MoR+Skip replay actions must have shape "
                f"[batch, sequence, {inner_count}]"
            )
        if mode == "budget":
            if target_skip_densities is None or target_skip_densities.shape != (batch,):
                raise ValueError(
                    "budget MoR+Skip routing requires target_skip_densities shaped [batch]"
                )
            target_skip_densities = target_skip_densities.to(
                device=token_ids.device, dtype=torch.float32
            )
            if bool(((target_skip_densities < 0) | (target_skip_densities > 1)).any()):
                raise ValueError("target_skip_densities must lie in [0,1]")
            epsilon = torch.as_tensor(
                exploration_epsilon, device=token_ids.device, dtype=torch.float32
            )
            if epsilon.ndim == 0:
                epsilon = epsilon.expand(batch)
            if epsilon.shape != (batch,) or bool(((epsilon < 0) | (epsilon > 1)).any()):
                raise ValueError("exploration_epsilon must be scalar or [batch] in [0,1]")

        previous_selected = torch.ones(
            batch, seq_len, device=token_ids.device, dtype=torch.bool
        )
        recursion_hard, recursion_soft = [], []
        auxiliary_losses, accuracies = [], []
        skip_gates = [x.new_zeros((batch, seq_len)) for _ in range(inner_count)]
        skip_soft = [x.new_zeros((batch, seq_len)) for _ in range(inner_count)]
        skip_actions = [
            torch.zeros((batch, seq_len), device=x.device, dtype=torch.long)
            for _ in range(inner_count)
        ]
        skip_log_probs = [x.new_zeros((batch, seq_len)) for _ in range(inner_count)]
        skip_behavior_log_probs = [
            x.new_zeros((batch, seq_len)) for _ in range(inner_count)
        ]
        skip_entropies = [x.new_zeros((batch, seq_len)) for _ in range(inner_count)]
        skip_logits = [x.new_zeros((batch, seq_len, 2)) for _ in range(inner_count)]
        skip_decision_masks = [
            torch.zeros((batch, seq_len), device=x.device, dtype=torch.bool)
            for _ in range(inner_count)
        ]

        for recursion_index in range(self.config.recursion_steps):
            raw_logits = self.recursion_routers[recursion_index](x).squeeze(-1)
            policy_execute = raw_logits.sigmoid()
            candidates = previous_selected
            if recursion_index == 0:
                selected = candidates
            elif mode == "topk":
                capacity = self._capacity(recursion_index)
                count = max(1, int(capacity * seq_len))
                scores = raw_logits.masked_fill(~candidates, float("-inf"))
                selected = torch.zeros_like(candidates)
                for batch_index in range(batch):
                    take = min(count, int(candidates[batch_index].sum().item()))
                    if take:
                        indices = scores[batch_index].topk(take, sorted=False).indices
                        selected[batch_index, indices] = True
            else:
                # Hybrid GRPO changes only the inner SkipLayer policy. MoR
                # admission remains its deterministic deployable policy.
                selected = policy_execute.ge(0.5) & candidates

            capacity = self._capacity(recursion_index)
            target_count = max(1, int(capacity * seq_len))
            oracle_target = torch.zeros_like(candidates)
            scores = raw_logits.masked_fill(~candidates, float("-inf"))
            for batch_index in range(batch):
                take = min(target_count, int(candidates[batch_index].sum().item()))
                if take:
                    indices = scores[batch_index].topk(take, sorted=False).indices
                    oracle_target[batch_index, indices] = True
            if bool(candidates.any()):
                auxiliary_losses.append(
                    F.binary_cross_entropy_with_logits(
                        raw_logits[candidates], oracle_target[candidates].float()
                    )
                )
                accuracies.append(
                    policy_execute.ge(0.5)[candidates]
                    .eq(oracle_target[candidates])
                    .float()
                    .mean()
                )
            else:
                auxiliary_losses.append(raw_logits.sum() * 0.0)
                accuracies.append(raw_logits.new_ones(()))

            output = x.clone()
            for batch_index in range(batch):
                indices = selected[batch_index].nonzero(as_tuple=False).flatten()
                if indices.numel() == 0:
                    continue
                initial = x[batch_index : batch_index + 1, indices]
                transformed = initial
                for block_index, block in enumerate(self.blocks[1:-1]):
                    inner_index = (
                        recursion_index * self.config.recursion_block_layers + block_index
                    )
                    router_input = block.ln1(transformed)
                    local_logits = skip_router.forward_layer(router_input, inner_index)
                    supplied = (
                        actions[batch_index : batch_index + 1, indices, inner_index]
                        if actions is not None else None
                    )
                    if mode == "budget" and supplied is None:
                        scaled = local_logits / self.config.gumbel_temperature
                        policy_log = F.log_softmax(scaled, dim=-1)
                        probabilities = policy_log.exp()
                        controller = target_skip_densities[batch_index]
                        mixed_execute = (
                            (1.0 - epsilon[batch_index]) * probabilities[..., 1]
                            + epsilon[batch_index] * controller
                        )
                        behavior_execute = mixed_execute.clamp(
                            torch.finfo(local_logits.dtype).eps,
                            1.0 - torch.finfo(local_logits.dtype).eps,
                        )
                        if bool(epsilon[batch_index].eq(1.0)):
                            behavior_execute = torch.full_like(
                                behavior_execute, controller
                            )
                        chosen = torch.rand_like(behavior_execute).lt(
                            behavior_execute
                        ).long()
                        gate = chosen.to(local_logits.dtype)
                        soft = probabilities[..., 1]
                        log_probability = torch.where(
                            chosen.bool(), policy_log[..., 1], policy_log[..., 0]
                        )
                        behavior_log_probability = torch.where(
                            chosen.bool(),
                            behavior_execute.log(),
                            (-behavior_execute).log1p(),
                        )
                        entropy = -(probabilities * policy_log).sum(dim=-1)
                    else:
                        skip_mode = (
                            "gumbel" if mode == "topk" and self.training
                            else "sample" if mode == "sample"
                            else "greedy"
                        )
                        gate, soft, chosen, log_probability, entropy = route_from_logits(
                            local_logits,
                            skip_mode,
                            self.config.gumbel_temperature,
                            supplied,
                        )
                        behavior_log_probability = log_probability

                    if self.config.sparse_inference and not self.training and mode == "greedy":
                        transformed = block.forward_selected(transformed, chosen.bool())
                    else:
                        transformed = apply_block_gate(
                            transformed, block(transformed), gate.unsqueeze(-1)
                        )

                    for collection, local_value, fill in (
                        (skip_gates, gate, 0.0),
                        (skip_soft, soft, 0.0),
                        (skip_actions, chosen, 0),
                        (skip_log_probs, log_probability, 0.0),
                        (skip_behavior_log_probs, behavior_log_probability, 0.0),
                        (skip_entropies, entropy, 0.0),
                    ):
                        collection[inner_index][batch_index, indices] = local_value[0]
                    skip_logits[inner_index][batch_index, indices] = local_logits[0]
                    skip_decision_masks[inner_index][batch_index, indices] = True

                weight = (
                    self.config.mor_router_alpha
                    * policy_execute[batch_index, indices]
                ).view(1, -1, 1)
                output[batch_index, indices] = (
                    initial + weight * (transformed - initial)
                )[0]
            x = output
            previous_selected = selected
            recursion_hard.append(selected.float())
            recursion_soft.append(policy_execute * candidates.float())

        x = self.blocks[-1](x)
        hard_inner = torch.stack(skip_gates, dim=-1)
        soft_inner = torch.stack(skip_soft, dim=-1)
        decision_mask = torch.stack(skip_decision_masks, dim=-1)
        ones = torch.ones(batch, seq_len, device=x.device, dtype=x.dtype)
        hard_layers = torch.cat((ones.unsqueeze(-1), hard_inner, ones.unsqueeze(-1)), dim=-1)
        soft_layers = torch.cat((ones.unsqueeze(-1), soft_inner, ones.unsqueeze(-1)), dim=-1)
        recursion_hard_tensor = torch.stack(recursion_hard, dim=-1)
        recursion_soft_tensor = torch.stack(recursion_soft, dim=-1)
        return self.finish(
            x,
            targets,
            soft_gates=soft_layers,
            hard_gates=hard_layers,
            actions=torch.stack(skip_actions, dim=-1),
            action_log_probs=torch.stack(skip_log_probs, dim=-1),
            behavior_log_probs=torch.stack(skip_behavior_log_probs, dim=-1),
            routing_entropy=torch.stack(skip_entropies, dim=-1),
            route_logits=torch.stack(skip_logits, dim=2),
            routing_decision_mask=decision_mask,
            mor_aux_loss=torch.stack(auxiliary_losses).sum(),
            mor_router_accuracy=torch.stack(accuracies).mean(),
            recursion_utilization=recursion_hard_tensor.mean(dim=(0, 1)),
            recursion_soft_utilization=recursion_soft_tensor.mean(dim=(0, 1)),
            mor_actions=recursion_hard_tensor.long(),
            skip_hard_gates=hard_inner,
            skip_soft_gates=soft_inner,
            skip_conditional_utilization=self._conditional_mean(
                hard_inner, decision_mask
            ),
            skip_soft_conditional_utilization=self._conditional_mean(
                soft_inner, decision_mask
            ),
            combined_block_utilization=hard_inner.mean(dim=(0, 1)),
        )
DynamicDepthTransformer = SparseDepthTransformer


def build_model(config: ModelConfig) -> TransformerBase:
    if config.model_type == "dense":
        return DenseTransformer(config)
    if config.model_type == "mor":
        return MixtureOfRecursionsTransformer(config)
    if config.model_type == "mor_skip":
        return MixtureOfRecursionsSkipLayerTransformer(config)
    if config.model_type in {
        "sparse_moe_mtp", "sparse_moe_mtp_mla", "sparse_mtp_mla"
    }:
        return SparseMoEMTPTransformer(config)
    return SparseDepthTransformer(config)
