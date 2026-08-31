"""Device, reproducibility, metrics, and checkpoint utilities."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

from config import ModelConfig
from model import ModelOutput, TransformerBase, build_model


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
        entropy = output.routing_entropy.float().mean().item()
    density = hard.mean().item()
    return {
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


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1).eq(targets).float().mean()


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
