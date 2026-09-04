"""Token-difficulty versus effective-depth analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch

from tiny_deepseek.data.shakespeare import TinyShakespeareData
from tiny_deepseek.core.model import TransformerBase


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


@torch.inference_mode()
def difficulty_depth_analysis(
    model: TransformerBase,
    dataset: TinyShakespeareData,
    device: torch.device,
    batch_size: int,
    batches: int,
    analysis_seed: int = 54321,
) -> Dict[str, Any]:
    model.eval()
    losses, depths = [], []
    generator = torch.Generator().manual_seed(analysis_seed)
    for _ in range(batches):
        x, y = dataset.get_batch("val", batch_size, device, generator=generator)
        output = model(x, y, routing_mode="greedy")
        token_depth = (
            output.hard_gates.float().sum(dim=-1)
            if output.hard_gates is not None
            else torch.full_like(output.token_losses, float(model.config.n_layers))
        )
        losses.append(output.token_losses.cpu().reshape(-1))
        depths.append(token_depth.cpu().reshape(-1))
    nll = torch.cat(losses).numpy()
    depth = torch.cat(depths).numpy()
    pearson = float(np.corrcoef(nll, depth)[0, 1]) if np.std(depth) > 0 else float("nan")
    spearman = (
        float(np.corrcoef(_rankdata(nll), _rankdata(depth))[0, 1])
        if np.std(depth) > 0 else float("nan")
    )
    boundaries = np.quantile(nll, np.linspace(0, 1, 6))
    quantile_depth = []
    for index in range(5):
        mask = (nll >= boundaries[index]) & (
            nll <= boundaries[index + 1] if index == 4 else nll < boundaries[index + 1]
        )
        quantile_depth.append(float(depth[mask].mean()))
    return {
        "pearson_nll_depth": pearson,
        "spearman_nll_depth": spearman,
        "difficulty_quantile_depth": quantile_depth,
        "difficulty_quantile_labels": ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"],
        "tokens_analyzed": int(len(nll)),
    }


def save_difficulty_plot(result: Dict[str, Any], output_stem: str | Path) -> None:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(result["difficulty_quantile_labels"], result["difficulty_quantile_depth"], marker="o")
    axis.set_xlabel("Token NLL difficulty quantile")
    axis.set_ylabel("Average executed layers")
    axis.set_title("Token difficulty versus dynamic depth")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


@torch.inference_mode()
def token_skip_analysis(
    model: TransformerBase,
    dataset: TinyShakespeareData,
    device: torch.device,
    batch_size: int,
    batches: int,
    analysis_seed: int = 65432,
) -> Dict[str, Any]:
    """Paper Figure-7 analogue: average skipped layers for each character."""
    model.eval()
    counts = torch.zeros(dataset.vocab_size, dtype=torch.long)
    skipped = torch.zeros(dataset.vocab_size, dtype=torch.float64)
    generator = torch.Generator().manual_seed(analysis_seed)
    for _ in range(batches):
        x, y = dataset.get_batch("val", batch_size, device, generator=generator)
        output = model(x, y, routing_mode="greedy")
        token_skips = model.config.n_layers - output.hard_gates.float().sum(dim=-1)
        ids = x.detach().cpu().reshape(-1)
        counts.scatter_add_(0, ids, torch.ones_like(ids))
        skipped.scatter_add_(0, ids, token_skips.detach().cpu().double().reshape(-1))
    records = []
    for token_id in range(dataset.vocab_size):
        count = int(counts[token_id])
        if count:
            records.append(
                {
                    "token_id": token_id,
                    "token": dataset.itos[token_id],
                    "count": count,
                    "average_skipped_layers": float(skipped[token_id] / count),
                    "average_executed_layers": float(
                        model.config.n_layers - skipped[token_id] / count
                    ),
                }
            )
    by_skip = sorted(records, key=lambda row: row["average_skipped_layers"], reverse=True)
    return {
        "token_skip_records": records,
        "most_skipped_tokens": by_skip[:10],
        "least_skipped_tokens": list(reversed(by_skip[-10:])),
    }


def save_token_skip_artifacts(result: Dict[str, Any], output_stem: str | Path) -> None:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    records = result["token_skip_records"]
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)
    if not records:
        return
    frequencies = np.array([row["count"] for row in records], dtype=float)
    skipped = np.array([row["average_skipped_layers"] for row in records], dtype=float)
    fig, axis = plt.subplots(figsize=(8, 5))
    sizes = 30 + 170 * skipped / max(skipped.max(), 1e-9)
    axis.scatter(frequencies, skipped, s=sizes, alpha=0.55)
    for row in sorted(records, key=lambda item: item["average_skipped_layers"])[-8:]:
        label = repr(row["token"])[1:-1]
        axis.annotate(label, (row["count"], row["average_skipped_layers"]), fontsize=8)
    axis.set_xscale("log")
    axis.set_xlabel("Character frequency in validation sample (log scale)")
    axis.set_ylabel("Average skipped layers")
    axis.set_title("Token-wise SkipLayer behavior")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
