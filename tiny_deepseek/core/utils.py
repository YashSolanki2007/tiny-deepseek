"""Device, reproducibility, metrics, and checkpoint utilities."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

from tiny_deepseek.core.config import ModelConfig
from tiny_deepseek.core.model import ModelOutput, TransformerBase, build_model


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def routing_metrics(output: ModelOutput, n_layers: int) -> Dict[str, Any]:
    if output.hard_gates is None:
        hard = torch.ones(n_layers, device=output.logits.device)
        soft = hard
        entropy = 0.0
    else:
        hard = output.hard_gates.float().mean(dim=(0, 1))
        soft = output.soft_gates.float().mean(dim=(0, 1))
        if output.routing_decision_mask is not None and bool(output.routing_decision_mask.any()):
            entropy = output.routing_entropy.float()[output.routing_decision_mask].mean().item()
        else:
            entropy = output.routing_entropy.float().mean().item()
    density = hard.mean().item()
    metrics = {
        "mean_soft_gate": soft.mean().item(),
        "mean_hard_gate": density,
        "actual_density": density,
        "layers_per_token": density * n_layers,
        "compute_fraction": density,
        "skip_fraction": 1 - density,
        "routing_entropy": entropy,
        "layer_utilization": hard.detach().cpu().tolist(),
        "layer_soft_probability": soft.detach().cpu().tolist(),
    }
    if output.recursion_utilization is not None:
        metrics["recursion_utilization"] = (
            output.recursion_utilization.detach().float().cpu().tolist()
        )
        metrics["recursion_soft_utilization"] = (
            output.recursion_soft_utilization.detach().float().cpu().tolist()
        )
        metrics["mean_recursions_per_token"] = float(
            output.recursion_utilization.detach().float().sum().item()
        )
        metrics["mor_router_accuracy"] = float(
            output.mor_router_accuracy.detach().item()
        )
        metrics["mor_aux_loss"] = float(output.mor_aux_loss.detach().item())
    if output.skip_conditional_utilization is not None:
        metrics["skip_conditional_utilization"] = (
            output.skip_conditional_utilization.detach().float().cpu().tolist()
        )
        metrics["skip_soft_conditional_utilization"] = (
            output.skip_soft_conditional_utilization.detach().float().cpu().tolist()
        )
        metrics["combined_block_utilization"] = (
            output.combined_block_utilization.detach().float().cpu().tolist()
        )
        metrics["mean_conditional_skip_density"] = float(
            output.skip_conditional_utilization.detach().float().mean().item()
        )
    if output.moe_aux_loss is not None:
        utilization = output.expert_utilization.detach().float().cpu()
        affinity = output.expert_affinity.detach().float().cpu()
        metrics.update(
            {
                "moe_aux_loss": float(output.moe_aux_loss.detach().item()),
                "moe_router_entropy": float(output.moe_router_entropy.detach().item()),
                "expert_utilization": utilization.tolist(),
                "expert_affinity": affinity.tolist(),
                "expert_utilization_min": float(utilization.min().item()),
                "expert_utilization_max": float(utilization.max().item()),
                "expert_utilization_cv": float(
                    utilization.std(unbiased=False).item()
                    / max(utilization.mean().item(), 1e-9)
                ),
            }
        )
    if output.mtp_loss is not None:
        metrics["mtp_loss"] = float(output.mtp_loss.detach().item())
        metrics["mtp_accuracy"] = float(output.mtp_accuracy.detach().item())
    return metrics


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    valid = targets.ne(-100)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return logits.argmax(dim=-1).eq(targets)[valid].float().mean()


def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    gradients = [p.grad.detach().float().norm(2) for p in parameters if p.grad is not None]
    if not gradients:
        return 0.0
    return torch.stack(gradients).norm(2).item()


def estimate_dense_block_flops(config: ModelConfig, seq_len: int) -> float:
    d, ff = config.d_model, config.d_ff
    per_layer = 8 * seq_len * d * d + 4 * seq_len * seq_len * d + 4 * seq_len * d * ff
    return float(config.n_layers * per_layer)


def estimate_skiplayer_flops(
    config: ModelConfig, seq_len: int, density: float
) -> float:
    """Paper-style FLOPs including always-on key/value projections.

    Multiply-adds count as two operations, matching ``estimate_dense_block_flops``.
    The estimate excludes embeddings, layer norms, softmax, and the LM head.
    """
    d, ff, p = config.d_model, config.d_ff, float(density)
    key_value = 4 * seq_len * d * d
    active_query_output = 4 * p * seq_len * d * d
    active_attention = 4 * p * seq_len * seq_len * d
    active_ffn = 4 * p * seq_len * d * ff
    router = 4 * seq_len * d
    return float(
        config.n_layers
        * (key_value + active_query_output + active_attention + active_ffn + router)
    )


def estimate_sparse_moe_mtp_flops(
    config: ModelConfig, seq_len: int, density: float
) -> float:
    """Inference FLOPs for SkipLayer+MoE; the training-only MTP head is discarded."""
    base = estimate_skiplayer_flops(config, seq_len, density)
    expert_router = (
        2
        * config.n_layers
        * float(density)
        * seq_len
        * config.d_model
        * config.moe_num_experts
    )
    return float(base + expert_router)


def estimate_mla_skiplayer_moe_mtp_flops(
    config: ModelConfig, seq_len: int, density: float
) -> float:
    """Inference FLOPs for MLA, SkipLayer, and either dense or sparse FFNs."""
    d, heads, length, active = (
        config.d_model, config.n_heads, float(seq_len), float(density)
    )
    qk = config.mla_qk_nope_head_dim + config.mla_qk_rope_head_dim
    value = config.mla_v_head_dim
    rank = config.mla_kv_lora_rank
    if config.mla_q_lora_rank > 0:
        query = 2 * active * length * (
            d * config.mla_q_lora_rank
            + config.mla_q_lora_rank * heads * qk
        )
    else:
        query = 2 * active * length * d * heads * qk
    kv_down = 2 * length * d * (rank + config.mla_qk_rope_head_dim)
    kv_up = 2 * length * rank * heads * (
        config.mla_qk_nope_head_dim + value
    )
    attention = 2 * active * length * length * heads * (qk + value)
    output = 2 * active * length * heads * value * d
    feed_forward = 4 * active * length * d * config.d_ff
    skip_router = 4 * length * d
    expert_router = (
        2 * active * length * d * config.moe_num_experts
        if config.model_type in {"sparse_moe_mtp", "sparse_moe_mtp_mla"}
        else 0.0
    )
    return float(
        config.n_layers
        * (
            query + kv_down + kv_up + attention + output
            + feed_forward + skip_router + expert_router
        )
    )


def estimate_mor_flops(
    config: ModelConfig,
    seq_len: int,
    recursion_utilization: Iterable[float],
) -> float:
    """MoR forward FLOPs with Middle-Cycle and recursion-wise KV caching.

    Entry and exit layers process the full sequence. At each recursion, Q/K/V,
    FFN, and output projections scale linearly with the active token fraction,
    while attention matmuls scale quadratically because both query and KV
    sequences are restricted to the selected tokens.
    """
    utilization = [float(value) for value in recursion_utilization]
    if len(utilization) != config.recursion_steps:
        raise ValueError("recursion utilization length must match recursion_steps")
    d, ff, length = config.d_model, config.d_ff, float(seq_len)
    full_layer = 8 * length * d * d + 4 * length * length * d + 4 * length * d * ff
    total = 2 * full_layer
    previous = 1.0
    for active in utilization:
        p = min(max(active, 0.0), previous)
        projections = 8 * p * length * d * d
        attention = 4 * (p * length) ** 2 * d
        feed_forward = 4 * p * length * d * ff
        total += config.recursion_block_layers * (
            projections + attention + feed_forward
        )
        # One scalar linear router is evaluated for each candidate token.
        total += 2 * previous * length * d
        previous = p
    return float(total)


def estimate_mor_skip_flops(
    config: ModelConfig,
    seq_len: int,
    recursion_utilization: Iterable[float],
    combined_block_utilization: Iterable[float],
) -> float:
    """FLOPs for MoR admission followed by inner SkipLayer execution.

    For an admitted recursion, K/V projections retain every admitted token as
    context, matching SkipLayer semantics. Query/output projections, attention
    queries, and FFNs are evaluated only for tokens whose inner gate executes.
    """
    recursion = [float(value) for value in recursion_utilization]
    combined = [float(value) for value in combined_block_utilization]
    expected_inner = config.recursion_steps * config.recursion_block_layers
    if len(recursion) != config.recursion_steps:
        raise ValueError("recursion utilization length must match recursion_steps")
    if len(combined) != expected_inner:
        raise ValueError("combined utilization must match recursive block applications")
    d, ff, length = config.d_model, config.d_ff, float(seq_len)
    full_layer = 8 * length * d * d + 4 * length * length * d + 4 * length * d * ff
    total = 2 * full_layer
    previous = 1.0
    inner_index = 0
    for admitted in recursion:
        p = min(max(admitted, 0.0), previous)
        total += 2 * previous * length * d  # scalar MoR admission router
        for _ in range(config.recursion_block_layers):
            executed = min(max(combined[inner_index], 0.0), p)
            key_value = 4 * p * length * d * d
            active_query_output = 4 * executed * length * d * d
            attention = 4 * executed * p * length * length * d
            feed_forward = 4 * executed * length * d * ff
            skip_router = 4 * p * length * d
            total += (
                key_value + active_query_output + attention + feed_forward + skip_router
            )
            inner_index += 1
        previous = p
    return float(total)


def write_json(path: str | Path, values: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, default=str) + "\n", encoding="utf-8")


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # map_location may move serialized RNG byte tensors to CUDA/MPS, while
    # PyTorch's RNG restoration APIs require CPU ByteTensors.
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def checkpoint_payload(
    model: TransformerBase,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    step: int,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    training_config: Dict[str, Any],
    best_metrics: Dict[str, Any],
    summary: Optional[Dict[str, Any]] = None,
    reference_router_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    return {
        "checkpoint_version": 2,
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "step": step,
        "stoi": stoi,
        "itos": itos,
        "training_config": training_config,
        "best_metrics": best_metrics,
        "summary": summary,
        "rng_state": capture_rng_state(),
        "reference_router_state": reference_router_state,
    }


def save_checkpoint(path: str | Path, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(**kwargs), path)


def load_checkpoint(
    path: str | Path, device: torch.device | str
) -> tuple[TransformerBase, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_version", 1) < 2:
        raise ValueError(
            "This is a first-generation sigmoid-router checkpoint. Its saved metrics remain valid, "
            "but its one-logit router is not shape-compatible with the new two-class ST-Gumbel model."
        )
    model = build_model(ModelConfig.from_dict(checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def restore_training_state(
    checkpoint: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> int:
    if checkpoint.get("optimizer_state"):
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if checkpoint.get("scheduler_state"):
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    restore_rng_state(checkpoint.get("rng_state"))
    return int(checkpoint.get("step", 0))
