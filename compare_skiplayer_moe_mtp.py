"""Deterministically compare SkipLayer+GRPO with its MoE+MTP extension."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import TinyShakespeareData
from evaluation import evaluate_model
from model import SparseMoEMTPTransformer
from utils import (
    estimate_dense_block_flops,
    estimate_mla_skiplayer_moe_mtp_flops,
    estimate_skiplayer_flops,
    estimate_sparse_moe_mtp_flops,
    load_checkpoint,
    select_device,
    write_json,
)


def active_parameters(model, density: float) -> float:
    block_parameters = sum(p.numel() for block in model.blocks for p in block.parameters())
    router_parameters = sum(p.numel() for p in model.router.parameters())
    always_active = model.parameter_count() - block_parameters - router_parameters
    if isinstance(model, SparseMoEMTPTransformer):
        mtp_parameters = sum(
            p.numel()
            for module in (
                model.mtp_hidden_norm, model.mtp_token_norm, model.mtp_projection,
                model.mtp_block, model.mtp_output_norm,
            )
            for p in module.parameters()
        )
        always_active -= mtp_parameters
        experts = sum(
            p.numel() for block in model.blocks for expert in block.mlp.experts
            for p in expert.parameters()
        )
        block_parameters -= experts
        experts *= model.config.moe_top_k / model.config.moe_num_experts
        return always_active + router_parameters + density * (block_parameters + experts)
    return always_active + router_parameters + density * block_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default=(
            "experiments/mor_comparison_seed42/"
            "skiplayer_grpo_clip0p5_lam1_seed42/checkpoints/best_quality_compute.pt"
        ),
    )
    parser.add_argument(
        "--previous-mha",
        default=(
            "experiments/skiplayer_moe_mtp_seed42/"
            "grpo/checkpoints/best_quality_compute.pt"
        ),
    )
    parser.add_argument(
        "--proposed",
        default=(
            "experiments/skiplayer_moe_mtp_mla_rope_seed42/"
            "grpo/checkpoints/best_quality_compute.pt"
        ),
    )
    parser.add_argument(
        "--results-dir", default="results/skiplayer_moe_mtp_mla_rope_seed42"
    )
    parser.add_argument("--data-path", default="data/input.txt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = select_device(args.device)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    experiments = (
        ("SkipLayer + GRPO", args.baseline),
        ("SkipLayer + GRPO + MoE + MTP (MHA + learned positions)", args.previous_mha),
        ("SkipLayer + GRPO + MoE + MTP (MLA + RoPE)", args.proposed),
    )
    for name, path in experiments:
        model, checkpoint = load_checkpoint(path, device)
        dataset = TinyShakespeareData(args.data_path, model.config.context_length)
        if dataset.stoi != checkpoint["stoi"]:
            raise ValueError(f"vocabulary mismatch for {path}")
        metrics = evaluate_model(
            model, dataset, device, args.batch_size, args.eval_iters,
            target_density=0.5, eval_seed=12345,
        )
        dense = estimate_dense_block_flops(model.config, model.config.context_length)
        if model.config.attention_type == "mla":
            flops = estimate_mla_skiplayer_moe_mtp_flops(
                model.config, model.config.context_length, metrics["compute_fraction"]
            )
        elif isinstance(model, SparseMoEMTPTransformer):
            flops = estimate_sparse_moe_mtp_flops(
                model.config, model.config.context_length, metrics["compute_fraction"]
            )
        else:
            flops = estimate_skiplayer_flops(
                model.config, model.config.context_length, metrics["compute_fraction"]
            )
        rows.append(
            {
                "model": name, "checkpoint": str(path),
                "attention_type": model.config.attention_type,
                "position_embedding_type": model.config.position_embedding_type,
                "val_loss": metrics["val_loss"], "val_perplexity": metrics["val_perplexity"],
                "val_accuracy": metrics["val_accuracy"],
                "layers_per_token": metrics["layers_per_token"],
                "skip_fraction": metrics["skip_fraction"],
                "estimated_flops_vs_full_dense": flops / dense,
                "parameter_count": model.parameter_count(),
                "inference_parameter_count": (
                    model.parameter_count()
                    - sum(
                        p.numel()
                        for module in (
                            model.mtp_hidden_norm, model.mtp_token_norm,
                            model.mtp_projection, model.mtp_block, model.mtp_output_norm,
                        )
                        for p in module.parameters()
                    )
                    if isinstance(model, SparseMoEMTPTransformer)
                    else model.parameter_count()
                ),
                "active_parameter_estimate": active_parameters(model, metrics["compute_fraction"]),
                "mtp_loss": metrics.get("mtp_loss"), "mtp_accuracy": metrics.get("mtp_accuracy"),
                "moe_router_entropy": metrics.get("moe_router_entropy"),
                "expert_utilization_cv": metrics.get("expert_utilization_cv"),
            }
        )
        if isinstance(model, SparseMoEMTPTransformer):
            utilization = np.asarray(metrics["expert_utilization"])
            fig, axis = plt.subplots(figsize=(9, 5))
            image = axis.imshow(utilization, aspect="auto", vmin=0, cmap="viridis")
            axis.set_xlabel("Expert")
            axis.set_ylabel("Layer")
            axis.set_xticks(range(model.config.moe_num_experts))
            axis.set_yticks(range(model.config.n_layers))
            axis.set_title("Final expert utilization (share of routed assignments)")
            fig.colorbar(image, ax=axis, label="Utilization")
            fig.tight_layout()
            attention_tag = model.config.attention_type
            fig.savefig(results_dir / f"expert_utilization_{attention_tag}.png", dpi=180)
            fig.savefig(results_dir / f"expert_utilization_{attention_tag}.pdf")
            plt.close(fig)

    with (results_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    baseline, previous_mha, proposed = rows
    comparison = {
        "evaluation": {
            "dataset": "Tiny Shakespeare", "eval_seed": 12345,
            "eval_iters": args.eval_iters, "batch_size": args.batch_size,
        },
        "rows": rows,
        "proposed_minus_baseline": {
            key: proposed[key] - baseline[key]
            for key in (
                "val_loss", "val_perplexity", "val_accuracy", "layers_per_token",
                "skip_fraction", "estimated_flops_vs_full_dense", "parameter_count",
                "active_parameter_estimate",
            )
        },
        "proposed_minus_previous_mha": {
            key: proposed[key] - previous_mha[key]
            for key in (
                "val_loss", "val_perplexity", "val_accuracy", "layers_per_token",
                "skip_fraction", "estimated_flops_vs_full_dense", "parameter_count",
                "active_parameter_estimate",
            )
        },
    }
    write_json(results_dir / "comparison.json", comparison)
    fig, axis = plt.subplots(figsize=(7, 5))
    for row in rows:
        axis.scatter(row["estimated_flops_vs_full_dense"], row["val_loss"], s=90)
        axis.annotate(row["model"], (row["estimated_flops_vs_full_dense"], row["val_loss"]), xytext=(7, 5), textcoords="offset points")
    axis.set_xlabel("Estimated inference FLOPs vs 8-layer dense")
    axis.set_ylabel("Validation cross-entropy (lower is better)")
    axis.set_title("Quality/compute comparison")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(results_dir / "quality_compute.png", dpi=180)
    fig.savefig(results_dir / "quality_compute.pdf")
    plt.close(fig)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
