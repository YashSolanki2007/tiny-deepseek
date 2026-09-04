"""Generate per-experiment diagnostics and cross-experiment comparison plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def numeric(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def short_label(item: dict) -> str:
    if item.get("model") == "dense":
        return f"Full dense L{item.get('n_layers') or 8}"
    if item.get("model") == "mor":
        return "MoR + GRPO" if item.get("training_method") == "grpo" else "MoR"
    if item.get("model") == "mor_skip":
        return (
            "MoR + SkipLayer + GRPO"
            if item.get("training_method") == "grpo"
            else "MoR + SkipLayer"
        )
    if item.get("training_method") == "grpo":
        return "SkipLayer + GRPO"
    return "Supervised SkipLayer"


def read_log(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_series(rows: list[dict], keys: list[tuple[str, str]], title: str, ylabel: str, stem: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for key, label in keys:
        points = [(numeric(row.get("step")), numeric(row.get(key))) for row in rows]
        points = [(x, y) for x, y in points if not math.isnan(x) and not math.isnan(y)]
        if points:
            axis.plot([p[0] for p in points], [p[1] for p in points], label=label)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    axis.set_xlabel("Training step")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    if len(keys) > 1:
        axis.legend()
    save_figure(fig, stem)


def per_experiment_plots(experiment: Path) -> None:
    log_path = experiment / "training_metrics.csv"
    if not log_path.exists():
        return
    rows = read_log(log_path)
    plots = experiment / "plots"
    plot_series(rows, [("train_loss", "train")], "Training cross entropy", "Cross entropy", plots / "training_ce")
    plot_series(rows, [("val_loss", "validation")], "Validation cross entropy", "Cross entropy", plots / "validation_ce")
    plot_series(rows, [("val_perplexity", "validation")], "Validation perplexity", "Perplexity", plots / "validation_perplexity")
    plot_series(rows, [("train_accuracy", "train"), ("val_accuracy", "validation")], "Next-character accuracy", "Top-1 accuracy", plots / "accuracy")
    plot_series(rows, [("compute_fraction", "hard density"), ("target_density", "target")], "Executed density", "Fraction", plots / "density")
    plot_series(rows, [("layers_per_token", "effective depth")], "Effective depth", "Layers/token", plots / "effective_depth")
    plot_series(rows, [("mean_reward", "mean reward")], "GRPO reward", "Reward", plots / "grpo_reward")
    plot_series(rows, [("mean_ce_component", "CE"), ("mean_compute_penalty", "compute"), ("mean_kl", "KL")], "GRPO reward components", "Component value", plots / "grpo_components")
    plot_series(rows, [("routing_entropy", "entropy")], "Routing entropy", "Entropy (nats)", plots / "routing_entropy")
    plot_series(rows, [("mor_aux_loss", "auxiliary BCE")], "MoR auxiliary router loss", "BCE", plots / "mor_aux_loss")
    plot_series(rows, [("mor_router_accuracy", "threshold accuracy")], "MoR router sampling accuracy", "Accuracy", plots / "mor_router_accuracy")
    plot_series(rows, [("estimated_flops_vs_full_dense", "relative FLOPs")], "Estimated FLOPs", "Fraction of full dense", plots / "estimated_flops")

    recursion_keys = sorted(
        [key for key in rows[0] if key.startswith("recursion_") and key.endswith("_utilization") and "soft" not in key],
        key=lambda key: int(key.split("_")[1]),
    ) if rows else []
    if recursion_keys:
        plot_series(
            rows,
            [(key, f"recursion {key.split('_')[1]}") for key in recursion_keys],
            "Recursion utilization",
            "Active token fraction",
            plots / "recursion_utilization",
        )

    mor_admission_keys = sorted(
        [
            key for key in rows[0]
            if key.startswith("mor_recursion_") and key.endswith("_admission")
            and "soft" not in key
        ],
        key=lambda key: int(key.split("_")[2]),
    ) if rows else []
    if mor_admission_keys:
        plot_series(
            rows,
            [(key, f"recursion {key.split('_')[2]}") for key in mor_admission_keys],
            "Outer MoR admission",
            "Admitted token fraction",
            plots / "mor_admission",
        )
    conditional_skip_keys = sorted(
        [
            key for key in rows[0]
            if key.startswith("skip_r") and key.endswith("_conditional_execute")
            and "soft" not in key
        ]
    ) if rows else []
    if conditional_skip_keys:
        plot_series(
            rows,
            [(key, key.removeprefix("skip_").removesuffix("_conditional_execute")) for key in conditional_skip_keys],
            "Inner SkipLayer execution conditional on MoR admission",
            "Conditional execute fraction",
            plots / "inner_skip_conditional_utilization",
        )
    combined_keys = sorted(
        [
            key for key in rows[0]
            if key.startswith("combined_r") and key.endswith("_utilization")
        ]
    ) if rows else []
    if combined_keys:
        plot_series(
            rows,
            [(key, key.removeprefix("combined_").removesuffix("_utilization")) for key in combined_keys],
            "Combined MoR × SkipLayer utilization",
            "Executed token fraction",
            plots / "combined_block_utilization",
        )

    budget_depth_keys = sorted(
        [
            key for key in rows[0]
            if key.startswith("budget_") and key.endswith("_depth")
            and key.split("_")[1].isdigit()
        ],
        key=lambda key: int(key.split("_")[1]),
    ) if rows else []
    if budget_depth_keys:
        plot_series(
            rows,
            [(key, key.split("_")[1] + "-layer budget") for key in budget_depth_keys],
            "Achieved rollout depths",
            "Layers/token",
            plots / "budget_rollout_depths",
        )
        plot_series(
            rows,
            [(key.replace("_depth", "_ce"), key.split("_")[1] + "-layer budget") for key in budget_depth_keys],
            "Cross entropy by rollout budget",
            "Cross entropy",
            plots / "budget_rollout_ce",
        )

    hybrid_budget_keys = sorted(
        [
            key for key in rows[0]
            if key.startswith("budget_P") and key.endswith("_effective_depth")
        ]
    ) if rows else []
    if hybrid_budget_keys:
        plot_series(
            rows,
            [(key, key.split("_")[1] + " inner budget") for key in hybrid_budget_keys],
            "Hybrid GRPO achieved depths",
            "Layers/token",
            plots / "hybrid_budget_rollout_depths",
        )
        plot_series(
            rows,
            [(key.replace("_effective_depth", "_ce"), key.split("_")[1]) for key in hybrid_budget_keys],
            "Hybrid GRPO cross entropy by budget",
            "Cross entropy",
            plots / "hybrid_budget_rollout_ce",
        )
        plot_series(
            rows,
            [(key.replace("_effective_depth", "_flops_vs_full_dense"), key.split("_")[1]) for key in hybrid_budget_keys],
            "Hybrid GRPO FLOPs by budget",
            "Fraction of full dense",
            plots / "hybrid_budget_rollout_flops",
        )

    summary_path = experiment / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        utilization = summary.get("layer_utilization")
        if utilization:
            fig, axis = plt.subplots(figsize=(7, 4.5))
            axis.bar(range(len(utilization)), utilization)
            axis.set_xlabel("Transformer layer")
            axis.set_ylabel("Hard utilization")
            axis.set_ylim(0, 1.05)
            axis.set_title("Per-layer utilization")
            save_figure(fig, plots / "per_layer_utilization")

    layer_keys = sorted(
        [key for key in rows[0] if key.startswith("layer_") and key.endswith("_density")],
        key=lambda key: int(key.split("_")[1]),
    ) if rows else []
    eval_rows = [row for row in rows if row.get("split") == "validation"]
    if layer_keys and eval_rows:
        matrix = np.array([[numeric(row.get(key)) for key in layer_keys] for row in eval_rows]).T
        fig, axis = plt.subplots(figsize=(8, 4.5))
        image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        axis.set_xlabel("Validation checkpoint")
        axis.set_ylabel("Layer")
        axis.set_yticks(range(len(layer_keys)))
        fig.colorbar(image, ax=axis, label="Utilization")
        axis.set_title("Layer utilization over training")
        save_figure(fig, plots / "layer_utilization_heatmap")


def comparison_plots(experiments_dir: Path, results_dir: Path) -> None:
    summaries = []
    for path in experiments_dir.glob("*/summary.json"):
        values = json.loads(path.read_text())
        values["name"] = path.parent.name
        summaries.append(values)
    if not summaries:
        return
    paper_only = all(bool(item.get("paper_reproduction")) for item in summaries)
    dense_flops = next(
        (
            float(item["estimated_executed_block_flops_per_sequence"])
            for item in summaries
            if item.get("model") == "dense"
            and item.get("estimated_executed_block_flops_per_sequence") is not None
        ),
        math.nan,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    histories = []
    for experiment in sorted(experiments_dir.iterdir() if experiments_dir.exists() else []):
        log_path = experiment / "training_metrics.csv"
        if log_path.exists():
            histories.append((experiment.name, read_log(log_path)))
    for key, fallback, ylabel, name in (
        ("train_loss", "mean_ce_component", "Training cross entropy", "comparison_training_ce"),
        ("val_loss", None, "Validation cross entropy", "comparison_validation_ce"),
        ("val_perplexity", None, "Validation perplexity", "comparison_validation_perplexity"),
        ("val_accuracy", None, "Validation top-1 accuracy", "comparison_validation_accuracy"),
        ("compute_fraction", None, "Hard compute fraction", "comparison_density"),
        ("layers_per_token", None, "Average layers/token", "comparison_effective_depth"),
    ):
        fig, axis = plt.subplots(figsize=(8.5, 5.3))
        plotted = False
        for label, rows in histories:
            points = []
            for row in rows:
                value = numeric(row.get(key))
                if math.isnan(value) and fallback:
                    value = numeric(row.get(fallback))
                step = numeric(row.get("step"))
                if not math.isnan(step) and not math.isnan(value):
                    points.append((step, value))
            if points:
                axis.plot([point[0] for point in points], [point[1] for point in points], label=label)
                plotted = True
        if plotted:
            axis.set_xlabel("Training step")
            axis.set_ylabel(ylabel)
            axis.set_title(ylabel + " across experiments")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
            save_figure(fig, results_dir / name)
        else:
            plt.close(fig)
    for y_key, y_label, name in (
        ("val_loss", "Validation cross entropy", "pareto_ce"),
        ("val_perplexity", "Validation perplexity", "pareto_perplexity"),
    ):
        fig, axis = plt.subplots(figsize=(8, 5.5))
        points = []
        for item in summaries:
            if item.get(y_key) is None:
                continue
            if paper_only:
                flops = item.get("estimated_executed_block_flops_per_sequence")
                if flops is None or not math.isfinite(dense_flops):
                    continue
                compute = float(flops) / dense_flops
            else:
                if item.get("compute_fraction") is None:
                    continue
                compute = float(item["compute_fraction"])
            axis.scatter(compute, item[y_key], s=55)
            axis.annotate(short_label(item), (compute, item[y_key]), xytext=(5, 4), textcoords="offset points", fontsize=8)
            points.append((compute, float(item[y_key])))
        frontier, best_quality = [], float("inf")
        for compute, quality in sorted(points):
            if quality < best_quality:
                frontier.append((compute, quality))
                best_quality = quality
        if len(frontier) > 1:
            axis.plot(
                [point[0] for point in frontier], [point[1] for point in frontier],
                color="black", linestyle="--", alpha=0.55, label="observed non-dominated frontier",
            )
            axis.legend(fontsize=8)
        axis.set_xlabel(
            "Estimated block FLOPs relative to dense"
            if paper_only else "Hard compute fraction"
        )
        axis.set_ylabel(y_label)
        axis.set_title("Quality–compute Pareto comparison")
        axis.grid(alpha=0.25)
        save_figure(fig, results_dir / name)

    for key, ylabel, name in (
        ("training_time_sec", "Total training seconds", "training_time"),
        ("seconds_per_step", "Training seconds/step", "training_seconds_per_step"),
        ("tokens_per_sec", "Training tokens/second", "training_throughput"),
        ("generation_tokens_per_sec", "Generation tokens/second", "inference_speed"),
        ("generation_ms_per_token", "Generation milliseconds/token", "inference_latency"),
    ):
        available = [(item["name"], item.get(key)) for item in summaries if item.get(key) is not None]
        if not available:
            continue
        fig, axis = plt.subplots(figsize=(max(7, len(available) * 0.8), 4.5))
        axis.bar([x[0] for x in available], [x[1] for x in available])
        axis.tick_params(axis="x", rotation=35)
        axis.set_ylabel(ylabel)
        axis.set_title(ylabel)
        save_figure(fig, results_dir / name)

    grpo = [item for item in summaries if item.get("training_method") == "grpo" and item.get("before_grpo")]
    if grpo:
        fig, axis = plt.subplots(figsize=(7, 5))
        for item in grpo:
            before = item["before_grpo"]
            xs = [before["compute_fraction"], item["compute_fraction"]]
            ys = [before["val_loss"], item["val_loss"]]
            axis.plot(xs, ys, marker="o", label=short_label(item))
        axis.set_xlabel("Hard compute fraction")
        axis.set_ylabel("Validation CE")
        axis.set_title("Supervised → GRPO Pareto movement")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.25)
        save_figure(fig, results_dir / "sl_vs_grpo")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", default="artifacts/experiments")
    parser.add_argument("--results-dir", default="artifacts/results")
    args = parser.parse_args()
    experiments = Path(args.experiments_dir)
    for path in experiments.iterdir() if experiments.exists() else []:
        if path.is_dir():
            per_experiment_plots(path)
    comparison_plots(experiments, Path(args.results_dir))
    print(f"plots written below {args.results_dir} and each experiment/plots directory")


if __name__ == "__main__":
    main()
