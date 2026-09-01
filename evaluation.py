"""Shared deterministic evaluation for supervised and GRPO stages."""

from __future__ import annotations

import time
from typing import Any, Dict

import torch

from data import TinyShakespeareData
from losses import density_loss
from model import TransformerBase
from utils import perplexity, routing_metrics, synchronize_device, top1_accuracy


@torch.inference_mode()
def evaluate_model(
    model: TransformerBase,
    dataset: TinyShakespeareData,
    device: torch.device,
    batch_size: int,
    eval_iters: int,
    target_density: float = 1.0,
    eval_seed: int = 12345,
    routing_mode: str = "greedy",
) -> Dict[str, Any]:
    was_training = model.training
    model.eval()
    totals = {
        "val_loss": 0.0,
        "val_accuracy": 0.0,
        "density_loss": 0.0,
        "mean_soft_gate": 0.0,
        "mean_hard_gate": 0.0,
        "layers_per_token": 0.0,
        "compute_fraction": 0.0,
        "skip_fraction": 0.0,
        "routing_entropy": 0.0,
    }
    hard_layers = torch.zeros(model.config.n_layers)
    soft_layers = torch.zeros(model.config.n_layers)
    recursion_layers = (
        torch.zeros(model.config.recursion_steps)
        if model.config.model_type == "mor" else None
    )
    recursion_soft = torch.zeros_like(recursion_layers) if recursion_layers is not None else None
    mor_aux_loss = mor_router_accuracy = 0.0
    generator = torch.Generator().manual_seed(eval_seed)
    synchronize_device(device)
    started = time.perf_counter()
    for _ in range(eval_iters):
        x, y = dataset.get_batch("val", batch_size, device, generator=generator)
        output = model(x, y, routing_mode=routing_mode)
        route = routing_metrics(output, model.config.n_layers)
        totals["val_loss"] += output.lm_loss.item()
        totals["val_accuracy"] += top1_accuracy(output.logits, y).item()
        totals["density_loss"] += (
            density_loss(
                output.hard_gates,
                target_density,
                reduction="sum" if model.config.paper_reproduction else "mean",
            ).item()
            if output.hard_gates is not None
            else 0.0
        )
        for key in (
            "mean_soft_gate",
            "mean_hard_gate",
            "layers_per_token",
            "compute_fraction",
            "skip_fraction",
            "routing_entropy",
        ):
            totals[key] += route[key]
        hard_layers += torch.tensor(route["layer_utilization"])
        soft_layers += torch.tensor(route["layer_soft_probability"])
        if recursion_layers is not None:
            recursion_layers += torch.tensor(route["recursion_utilization"])
            recursion_soft += torch.tensor(route["recursion_soft_utilization"])
            mor_aux_loss += route["mor_aux_loss"]
            mor_router_accuracy += route["mor_router_accuracy"]
    synchronize_device(device)
    elapsed = time.perf_counter() - started
    for key in totals:
        totals[key] /= eval_iters
    totals["val_perplexity"] = perplexity(totals["val_loss"])
    totals["validation_time_sec"] = elapsed
    totals["layer_utilization"] = (hard_layers / eval_iters).tolist()
    totals["layer_soft_probability"] = (soft_layers / eval_iters).tolist()
    if recursion_layers is not None:
        totals["recursion_utilization"] = (recursion_layers / eval_iters).tolist()
        totals["recursion_soft_utilization"] = (recursion_soft / eval_iters).tolist()
        totals["mean_recursions_per_token"] = float(
            (recursion_layers / eval_iters).sum().item()
        )
        totals["mor_aux_loss"] = mor_aux_loss / eval_iters
        totals["mor_router_accuracy"] = mor_router_accuracy / eval_iters
    if was_training:
        model.train()
    return totals
