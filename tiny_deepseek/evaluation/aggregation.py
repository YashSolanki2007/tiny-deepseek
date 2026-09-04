"""Create per-seed and mean±std result tables without inventing missing values."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


COLUMNS = [
    "model", "router_type", "training_method", "seed", "target_density",
    "actual_density", "lambda_density", "lambda_grpo", "val_loss", "val_perplexity",
    "val_accuracy", "layers_per_token", "compute_fraction", "skip_fraction",
    "parameter_count", "active_parameter_estimate", "training_time_sec", "seconds_per_step", "tokens_per_sec",
    "generation_tokens_per_sec", "generation_ms_per_token", "mean_reward",
    "pearson_nll_depth", "spearman_nll_depth", "checkpoint", "source",
    "paper_reproduction", "n_layers", "effective_target_depth", "learning_rate",
    "estimated_executed_block_flops_per_sequence", "dense_block_flops_per_sequence",
    "grpo_variant", "depth_budgets", "exploration_epsilon", "ppo_clip_epsilon",
    "clip_fraction",
    "mean_probability_ratio",
    "mor_reproduction", "experiment_family", "recursion_steps",
    "recursion_block_layers", "mor_capacity_factors", "mor_aux_loss_coefficient",
    "recursion_utilization", "recursion_soft_utilization",
    "mean_recursions_per_token", "mor_router_accuracy", "mor_aux_loss",
    "estimated_flops_vs_full_dense",
    "skip_density_budgets", "mean_conditional_skip_density",
    "skip_conditional_utilization", "skip_soft_conditional_utilization",
    "combined_block_utilization", "source_mor_checkpoint",
    "mtp_loss", "mtp_accuracy", "moe_num_experts", "moe_top_k",
    "moe_aux_loss", "moe_router_entropy", "expert_utilization_min",
    "expert_utilization_max", "expert_utilization_cv", "source_checkpoint",
    "attention_type", "position_embedding_type", "mla_kv_lora_rank",
    "mla_qk_nope_head_dim", "mla_qk_rope_head_dim", "mla_v_head_dim",
]
GROUP_KEYS = ["model", "router_type", "training_method", "target_density", "lambda_density", "lambda_grpo"]
METRICS = [
    "val_loss", "val_perplexity", "val_accuracy", "layers_per_token", "compute_fraction",
    "skip_fraction", "training_time_sec", "seconds_per_step", "tokens_per_sec", "generation_tokens_per_sec",
]


def value_or_nan(values: dict, key: str):
    value = values.get(key)
    return float("nan") if value is None or value == "" else value


def grouping_value(values: dict, key: str):
    """Use a stable sentinel for missing keys so seeds aggregate together."""
    value = values.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", default="artifacts/experiments")
    parser.add_argument("--results-dir", default="artifacts/results")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(Path(args.experiments_dir).glob("*/summary.json")):
        values = json.loads(path.read_text())
        values.setdefault("actual_density", values.get("compute_fraction"))
        values.setdefault("router_type", values.get("router", "none"))
        values.setdefault("training_method", "supervised")
        values["source"] = str(path)
        rows.append(values)
    if not rows:
        raise SystemExit("No experiment summary.json files found; refusing to fabricate results")
    with (results_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value_or_nan(row, key) for key in COLUMNS})

    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(grouping_value(row, key) for key in GROUP_KEYS)].append(row)
    aggregate_columns = GROUP_KEYS + ["num_seeds"] + [f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")]
    with (results_dir / "aggregate_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_columns)
        writer.writeheader()
        for group, members in grouped.items():
            output = {
                key: (float("nan") if value is None else value)
                for key, value in zip(GROUP_KEYS, group)
            }
            output["num_seeds"] = len(members)
            for metric in METRICS:
                values = np.array([float(value_or_nan(member, metric)) for member in members], dtype=float)
                finite = values[np.isfinite(values)]
                output[f"{metric}_mean"] = float(finite.mean()) if len(finite) else float("nan")
                output[f"{metric}_std"] = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0 if len(finite) == 1 else float("nan")
            writer.writerow(output)
    print(f"aggregated {len(rows)} runs into {results_dir}")


if __name__ == "__main__":
    main()
