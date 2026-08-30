"""Training, checkpoint, and routing metric helpers."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Optional

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_compute_lambda(
    step: int,
    total_steps: int,
    target_lambda: float,
    warmup_start: float = 0.10,
    warmup_end: float = 0.30,
) -> float:
    progress = step / max(total_steps, 1)
    if progress < warmup_start:
        return 0.0
    if progress < warmup_end:
        width = max(warmup_end - warmup_start, 1e-12)
        return target_lambda * (progress - warmup_start) / width
    return target_lambda


def compute_penalty(
    soft_gates: Optional[torch.Tensor], mode: str, target_compute: float
) -> torch.Tensor:
    if soft_gates is None:
        raise ValueError("Compute penalties only apply to dynamic models")
    mean_probability = soft_gates.mean()
    if mode == "linear":
        return mean_probability
    if mode == "target":
        return (mean_probability - target_compute).square()
    raise ValueError(f"Unknown compute loss mode: {mode}")


def routing_metrics(output: ModelOutput, n_layers: int) -> Dict[str, Any]:
    if output.hard_gates is None:
        utilization = torch.ones(n_layers, device=output.logits.device)
        soft_utilization = utilization
    else:
        utilization = output.hard_gates.float().mean(dim=(0, 1))
        soft_utilization = output.soft_gates.float().mean(dim=(0, 1))
    compute_fraction = utilization.mean().item()
    return {
        "mean_soft_gate": soft_utilization.mean().item(),
        "mean_hard_gate": compute_fraction,
        "layers_per_token": compute_fraction * n_layers,
        "compute_fraction": compute_fraction,
        "skip_fraction": 1.0 - compute_fraction,
        "layer_utilization": utilization.detach().cpu().tolist(),
        "layer_soft_probability": soft_utilization.detach().cpu().tolist(),
    }


def save_checkpoint(
    path: str | Path,
    model: TransformerBase,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    train_config: Dict[str, Any],
    summary: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "stoi": stoi,
        "itos": itos,
        "train_config": train_config,
        "summary": summary,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path, device: torch.device | str
) -> tuple[TransformerBase, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def write_json(path: str | Path, values: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def estimate_dense_block_flops(config: ModelConfig, seq_len: int) -> float:
    """Approximate forward FLOPs for all blocks for one sequence (multiply-add = 2 FLOPs)."""
    d, ff, layers = config.d_model, config.d_ff, config.n_layers
    per_layer = 8 * seq_len * d * d + 4 * seq_len * seq_len * d + 4 * seq_len * d * ff
    return float(layers * per_layer)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
