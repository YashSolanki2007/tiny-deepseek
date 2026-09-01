"""Build evidence-bound Markdown and self-contained HTML research reports."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import mimetypes
from collections import defaultdict
from pathlib import Path
from typing import Any


RESULT_FIELDS = [
    ("val_loss", "Validation CE"), ("val_perplexity", "Perplexity"),
    ("val_accuracy", "Accuracy"), ("layers_per_token", "Layers/token"),
    ("compute_fraction", "Compute fraction"),
    ("training_time_sec", "Training seconds"),
    ("generation_tokens_per_sec", "Generation tokens/s"),
]


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean_std(values: list[Any]) -> tuple[float, float]:
    clean = [float(value) for value in values if finite(value)]
    if not clean:
        return math.nan, math.nan
    mean = sum(clean) / len(clean)
    if len(clean) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in clean) / (len(clean) - 1)
    return mean, math.sqrt(variance)


def display(mean: float, std: float, percent: bool = False) -> str:
    if not finite(mean):
        return "NaN"
    scale, suffix = (100, "%") if percent else (1, "")
    return f"{scale * mean:.3f} ± {scale * std:.3f}{suffix}"


def model_label(row: dict[str, Any]) -> str:
    if row.get("model") == "dense":
        return "Dense"
    if row.get("model") == "mor":
        return "MoR + GRPO" if row.get("training_method") == "grpo" else "MoR"
    if row.get("model") == "mor_skip":
        return (
            "MoR + SkipLayer + GRPO"
            if row.get("training_method") == "grpo"
            else "MoR + SkipLayer"
        )
    if row.get("model") == "sparse" and row.get("paper_reproduction"):
        return "SkipLayer + GRPO" if row.get("training_method") == "grpo" else "SkipLayer"
    router = str(row.get("router_type", "unknown")).upper()
    method = " + GRPO" if row.get("training_method") == "grpo" else ""
    density = row.get("target_density")
    density_label = f" P={float(density):.2f}" if finite(density) else ""
    lam = row.get("lambda_density") if not method else row.get("lambda_grpo")
    lambda_label = f", λ={float(lam):g}" if finite(lam) else ""
    return f"{router}{method}{density_label}{lambda_label}"


def group_rows(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("model"), row.get("router_type"), row.get("training_method"),
            row.get("target_density"), row.get("lambda_density"), row.get("lambda_grpo"),
        )
        groups[key].append(row)
    return [(model_label(members[0]), members) for members in groups.values()]


def main_table(groups: list[tuple[str, list[dict[str, Any]]]]) -> str:
    dense_members = next((members for label, members in groups if label == "Dense"), [])
    dense_ce, _ = mean_std([member.get("val_loss") for member in dense_members])
    dense_ppl, _ = mean_std([member.get("val_perplexity") for member in dense_members])
    headings = ["Model", "Seeds"] + [label for _, label in RESULT_FIELDS] + ["ΔCE vs dense", "ΔPPL vs dense"]
    lines = ["| " + " | ".join(headings) + " |", "|" + "---|" * len(headings)]
    for label, members in groups:
        cells = [label, str(len(members))]
        for key, _ in RESULT_FIELDS:
            avg, std = mean_std([member.get(key) for member in members])
            cells.append(display(avg, std, key in {"val_accuracy", "compute_fraction"}))
        model_ce, _ = mean_std([member.get("val_loss") for member in members])
        model_ppl, _ = mean_std([member.get("val_perplexity") for member in members])
        cells.extend([
            f"{model_ce - dense_ce:+.4f}" if finite(model_ce) and finite(dense_ce) else "NaN",
            f"{model_ppl - dense_ppl:+.4f}" if finite(model_ppl) and finite(dense_ppl) else "NaN",
        ])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def grpo_assessment(rows: list[dict[str, Any]]) -> str:
    comparisons = []
    for row in rows:
        before = row.get("before_grpo")
        if row.get("training_method") != "grpo" or not isinstance(before, dict):
            continue
        values = (before.get("val_loss"), before.get("compute_fraction"), row.get("val_loss"), row.get("compute_fraction"))
        if all(finite(value) for value in values):
            comparisons.append((float(row["val_loss"]) - float(before["val_loss"]), float(row["compute_fraction"]) - float(before["compute_fraction"])))
    if not comparisons:
        return "No completed GRPO run with paired before/after measurements was found, so H3 cannot yet be evaluated."
    qtol, ctol = 0.01, 0.01
    clear = sum((dq < -qtol and dc <= ctol) or (dc < -ctol and dq <= qtol) for dq, dc in comparisons)
    degraded = sum((dq > qtol and dc >= -ctol) or (dc > ctol and dq >= -qtol) for dq, dc in comparisons)
    if clear == len(comparisons):
        verdict = "GRPO clearly improved the measured Pareto frontier."
    elif clear:
        verdict = "GRPO produced a small or inconsistent improvement across the measured runs."
    elif degraded == len(comparisons):
        verdict = "GRPO degraded routing in the measured runs."
    else:
        verdict = "GRPO made no meaningful, consistent difference in the measured Pareto frontier."
    details = "; ".join(f"ΔCE={dq:+.4f}, Δcompute={dc:+.4f}" for dq, dc in comparisons)
    return f"{verdict} Paired changes: {details}."


def setup_text(rows: list[dict[str, Any]], experiments_dir: Path) -> str:
    configs = []
    for path in sorted(experiments_dir.glob("*/config.json")):
        try:
            configs.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            pass
    if not configs:
        return "No readable experiment configurations were found."
    model = configs[0].get("model", {})
    supervised = [item.get("training", {}) for item in configs if item.get("stage") == "supervised"]
    grpo = [item.get("grpo", {}) for item in configs if item.get("stage") == "grpo"]
    devices = sorted({str(item.get("device", "unknown")) for item in configs})
    seeds = sorted({str(row.get("seed")) for row in rows if row.get("seed") is not None})
    targets = sorted({float(row["target_density"]) for row in rows if finite(row.get("target_density"))})
    train, rl = supervised[0] if supervised else {}, grpo[0] if grpo else {}
    return (
        f"Tiny Shakespeare, character tokens, 90/10 train/validation split. The shared backbone has "
        f"{model.get('n_layers', 'unknown')} layers, width {model.get('d_model', 'unknown')}, "
        f"{model.get('n_heads', 'unknown')} attention heads, FFN width {model.get('d_ff', 'unknown')}, "
        f"and context {model.get('context_length', 'unknown')}. Supervised training used up to "
        f"{train.get('max_steps', 'unknown')} steps with batch size {train.get('batch_size', 'unknown')}. "
        f"Devices: {', '.join(devices)}. Seeds: {', '.join(seeds) or 'none'}. Density targets: "
        f"{', '.join(f'{value:.2f}' for value in targets) or 'none'}. GRPO group size: "
        f"{rl.get('group_size', 'not run')}; compute penalty: {rl.get('lambda_compute_grpo', 'not run')}; "
        f"KL coefficient: {rl.get('beta_kl', 'not run')}."
    )


def routing_observations(rows: list[dict[str, Any]]) -> str:
    sparse = [row for row in rows if row.get("model") == "sparse" and finite(row.get("compute_fraction"))]
    if not sparse:
        return "No sparse-model evaluation metrics are available."
    errors = [abs(float(row["compute_fraction"]) - float(row["target_density"])) for row in sparse if finite(row.get("target_density"))]
    density_note = f"Mean absolute target-density error was {sum(errors)/len(errors):.3f}." if errors else "Target-density error was unavailable."
    correlations = [float(row["spearman_nll_depth"]) for row in sparse if finite(row.get("spearman_nll_depth"))]
    difficulty_note = f"Mean Spearman correlation between token NLL and depth was {sum(correlations)/len(correlations):+.3f}." if correlations else "Difficulty–depth correlations were not yet evaluated."
    layer_means: dict[int, list[float]] = defaultdict(list)
    for row in sparse:
        for index, value in enumerate(row.get("layer_utilization") or []):
            if finite(value):
                layer_means[index].append(float(value))
    if layer_means:
        averages = {index: sum(values) / len(values) for index, values in layer_means.items()}
        highest, lowest = max(averages, key=averages.get), min(averages, key=averages.get)
        layer_note = f"Across available runs, layer {highest} was used most ({averages[highest]:.3f}) and layer {lowest} least ({averages[lowest]:.3f})."
    else:
        layer_note = "Per-layer utilization was unavailable."
    return f"{density_note} {difficulty_note} {layer_note} Token maps are stored with each experiment."


def fair_router_comparison(rows: list[dict[str, Any]]) -> str:
    supervised = [row for row in rows if row.get("training_method") == "supervised" and row.get("model") == "sparse"]
    notes = []
    for linear in [row for row in supervised if row.get("router_type") == "linear"]:
        candidates = [row for row in supervised if row.get("router_type") == "gru" and row.get("seed") == linear.get("seed") and row.get("target_density") == linear.get("target_density") and row.get("lambda_density") == linear.get("lambda_density")]
        if not candidates:
            continue
        gru = candidates[0]
        required = (linear.get("compute_fraction"), gru.get("compute_fraction"), linear.get("val_loss"), gru.get("val_loss"))
        if not all(finite(value) for value in required):
            continue
        dc = float(gru["compute_fraction"]) - float(linear["compute_fraction"])
        dq = float(gru["val_loss"]) - float(linear["val_loss"])
        relation = "matched closely" if abs(dc) <= 0.02 else "not compute-matched"
        notes.append(f"seed {linear.get('seed')}, P={float(linear['target_density']):.2f}: GRU−Linear ΔCE={dq:+.4f}, Δcompute={dc:+.4f} ({relation})")
    if not notes:
        return "No paired Linear/GRU result at the same seed, target, and density coefficient is available, so H2 cannot yet be judged fairly."
    return "Paired supervised comparisons: " + "; ".join(notes) + "."


def training_dynamics(experiments_dir: Path) -> str:
    notes = []
    for path in sorted(experiments_dir.glob("*/training_metrics.csv")):
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        validation = [row for row in rows if finite(row.get("val_loss"))]
        if validation:
            first, last = float(validation[0]["val_loss"]), float(validation[-1]["val_loss"])
            notes.append(f"{path.parent.name}: validation CE {first:.4f}→{last:.4f}")
        policy = [row for row in rows if finite(row.get("mean_reward"))]
        if policy:
            first, last = policy[0], policy[-1]
            reward_change = f"reward {float(first['mean_reward']):.4f}→{float(last['mean_reward']):.4f}"
            entropy_change = (
                f", entropy {float(first['routing_entropy']):.3f}→{float(last['routing_entropy']):.3f}"
                if finite(first.get("routing_entropy")) and finite(last.get("routing_entropy")) else ""
            )
            notes.append(f"{path.parent.name}: {reward_change}{entropy_change}")
    return "; ".join(notes) + "." if notes else "No training-history CSV was available for a dynamics assessment."


def conclusion_text(rows: list[dict[str, Any]]) -> str:
    dense_ce, _ = mean_std([row.get("val_loss") for row in rows if row.get("model") == "dense"])
    linear = [row for row in rows if row.get("router_type") == "linear" and finite(row.get("val_loss")) and finite(row.get("compute_fraction"))]
    if linear and finite(dense_ce):
        best_linear = min(linear, key=lambda row: float(row["val_loss"]) + 0.1 * float(row["compute_fraction"]))
        linear_note = (
            f"The selected linear point used {float(best_linear['compute_fraction']):.3f} compute "
            f"with ΔCE={float(best_linear['val_loss']) - dense_ce:+.4f} versus dense."
        )
    else:
        linear_note = "H1 cannot be evaluated because a dense/linear pair is missing."
    fair = fair_router_comparison(rows)
    return f"{linear_note} {fair} {grpo_assessment(rows)} These measured tradeoffs—not raw loss alone—determine whether scaling is justified."


def recommendation_text(rows: list[dict[str, Any]]) -> str:
    seeds = {row.get("seed") for row in rows if row.get("seed") is not None}
    if len(seeds) < 3:
        return "First complete the core four-way comparison with at least three seeds; the present seed count is insufficient for a stable ranking."
    sparse = [row for row in rows if row.get("model") == "sparse" and finite(row.get("compute_fraction")) and finite(row.get("target_density"))]
    if sparse:
        target_error = sum(abs(float(row["compute_fraction"]) - float(row["target_density"])) for row in sparse) / len(sparse)
        if target_error > 0.05:
            return f"Tune `lambda_density` before scaling: mean absolute target error is {target_error:.3f}, so router comparisons are not yet compute-matched."
    grpo = grpo_assessment(rows)
    if "degraded" in grpo:
        return "Abandon GRPO for now and test the supervised routers on a deeper model or larger dataset; every measured paired GRPO movement degraded routing."
    if "no meaningful" in grpo or "cannot yet" in grpo:
        return "Keep the supervised backbone fixed and sweep `lambda_compute_grpo` over 0.01, 0.05, 0.1, and 0.2 before considering larger-scale GRPO."
    return "Repeat the apparent Pareto improvement on a deeper model and larger character dataset before combining sparse depth with MoE."


def inferred_layers(row: dict[str, Any]) -> int | None:
    if finite(row.get("n_layers")):
        return int(row["n_layers"])
    if finite(row.get("layers_per_token")) and finite(row.get("compute_fraction")):
        density = float(row["compute_fraction"])
        if density > 0:
            return int(round(float(row["layers_per_token"]) / density))
    return None


def paper_main_table(rows: list[dict[str, Any]]) -> str:
    dense = [row for row in rows if row.get("model") == "dense"]
    dense_flops, _ = mean_std(
        [row.get("estimated_executed_block_flops_per_sequence") for row in dense]
    )
    headings = [
        "Model", "Seeds", "Total layers", "Validation CE", "Perplexity",
        "Accuracy", "Layers/token", "Skipped", "Estimated block FLOPs",
        "FLOPs vs dense", "Generation tokens/s",
    ]
    lines = ["| " + " | ".join(headings) + " |", "|" + "---|" * len(headings)]
    for label, members in group_rows(rows):
        ce, ce_std = mean_std([row.get("val_loss") for row in members])
        ppl, ppl_std = mean_std([row.get("val_perplexity") for row in members])
        acc, acc_std = mean_std([row.get("val_accuracy") for row in members])
        depth, depth_std = mean_std([row.get("layers_per_token") for row in members])
        skipped, skipped_std = mean_std([row.get("skip_fraction") for row in members])
        flops, flops_std = mean_std(
            [row.get("estimated_executed_block_flops_per_sequence") for row in members]
        )
        speed, speed_std = mean_std([row.get("generation_tokens_per_sec") for row in members])
        layer_values = [inferred_layers(row) for row in members]
        layers = str(layer_values[0]) if layer_values and all(v == layer_values[0] for v in layer_values) else "mixed"
        flops_relative = f"{flops / dense_flops:.3f}×" if finite(flops) and finite(dense_flops) else "NaN"
        lines.append(
            "| " + " | ".join(
                [
                    label, str(len(members)), layers, display(ce, ce_std),
                    display(ppl, ppl_std), display(acc, acc_std, True),
                    display(depth, depth_std), display(skipped, skipped_std, True),
                    f"{flops / 1e6:.1f}M ± {flops_std / 1e6:.1f}M" if finite(flops) else "NaN",
                    flops_relative, display(speed, speed_std),
                ]
            ) + " |"
        )
    return "\n".join(lines)


def paper_setup_text(rows: list[dict[str, Any]], experiments_dir: Path) -> str:
    configs = []
    for path in sorted(experiments_dir.glob("*/config.json")):
        try:
            configs.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            pass
    def clean(values: set[Any]) -> list[Any]:
        return sorted(value for value in values if value is not None)

    widths = clean({item.get("model", {}).get("d_model") for item in configs})
    heads = clean({item.get("model", {}).get("n_heads") for item in configs})
    contexts = clean({item.get("model", {}).get("context_length") for item in configs})
    batches = clean({item.get("training", {}).get("batch_size") for item in configs})
    steps = clean({item.get("training", {}).get("max_steps") for item in configs})
    lrs = clean({item.get("training", {}).get("learning_rate") for item in configs})
    seeds = sorted({row.get("seed") for row in rows})
    return (
        "Tiny Shakespeare with character tokenization and a contiguous 90/10 split. "
        f"Widths: {widths}; attention heads: {heads}; contexts: {contexts}; batch sizes: {batches}; "
        f"training steps: {steps}; Adafactor learning rates: {lrs}; seeds: {seeds}. "
        "The attention head dimension is 64 and FFN width is 8× model width in the core protocol."
    )


def paper_conclusion(rows: list[dict[str, Any]]) -> str:
    dense = [row for row in rows if row.get("model") == "dense" and finite(row.get("val_loss"))]
    sparse = [row for row in rows if row.get("model") == "sparse" and finite(row.get("val_loss"))]
    if not dense or not sparse:
        return "A matched dense/SkipLayer pair is not yet complete."
    dense_ce, _ = mean_std([row["val_loss"] for row in dense])
    dense_flops, _ = mean_std(
        [row.get("estimated_executed_block_flops_per_sequence") for row in dense]
    )
    best = min(sparse, key=lambda row: float(row["val_loss"]))
    layers = inferred_layers(best) or 0
    target = float(best.get("target_density", math.nan))
    target_depth = layers * target if finite(target) else math.nan
    actual_depth = float(best.get("layers_per_token", math.nan))
    sparse_flops = float(best.get("estimated_executed_block_flops_per_sequence", math.nan))
    return (
        f"The selected SkipLayer point changed CE by {float(best['val_loss']) - dense_ce:+.4f} "
        f"relative to dense and executed {actual_depth:.3f} layers/token versus a "
        f"{target_depth:.3f} target. Its paper-style estimated block FLOPs were "
        f"{sparse_flops / dense_flops:.3f}× dense because key/value projections remain active at "
        "every physical layer. This is a matched-effective-depth comparison, not a claim of equal FLOPs."
    )


def paper_grpo_text(rows: list[dict[str, Any]]) -> str:
    grpo_rows = [row for row in rows if row.get("training_method") == "grpo"]
    if not grpo_rows:
        return "GRPO was not run in this result set."
    details = []
    for row in sorted(grpo_rows, key=lambda value: float(value.get("lambda_grpo", 0))):
        before = row.get("before_grpo") or {}
        if not all(
            finite(value)
            for value in (
                before.get("val_loss"), before.get("layers_per_token"),
                row.get("val_loss"), row.get("layers_per_token"),
            )
        ):
            continue
        details.append(
            f"λ={float(row['lambda_grpo']):g}: depth "
            f"{float(before['layers_per_token']):.3f}→{float(row['layers_per_token']):.3f}, "
            f"ΔCE={float(row['val_loss']) - float(before['val_loss']):+.4f}"
        )
    method = (
        "Each group used explicit low/medium/full depth budgets. Non-anchor actions were sampled "
        "from a recorded mixture of the router policy and a remaining-budget controller, and PPO "
        "ratios were computed at every token-layer decision. The deterministic full-depth rollout "
        "served as a quality reference but was excluded from the policy-gradient loss."
    )
    paired = []
    for row in grpo_rows:
        before = row.get("before_grpo") or {}
        if all(
            finite(value)
            for value in (
                before.get("val_loss"), row.get("val_loss"), row.get("layers_per_token")
            )
        ):
            paired.append(
                (
                    float(row["val_loss"]) - float(before["val_loss"]),
                    float(row["layers_per_token"]),
                )
            )
    criterion = ""
    if paired:
        eligible = [depth for delta_ce, depth in paired if delta_ce <= 0.02]
        reached = any(depth <= 4.0 and delta_ce <= 0.02 for delta_ce, depth in paired)
        if eligible:
            criterion = f" The lowest selected depth within ΔCE≤0.02 was {min(eligible):.3f}/8."
        criterion += (
            " The joint target of depth≤4 and ΔCE≤0.02 was reached."
            if reached
            else " The joint target of depth≤4 and ΔCE≤0.02 was not reached."
        )
    dense = [
        row for row in rows
        if row.get("model") == "dense"
        and finite(row.get("val_loss"))
        and finite(row.get("estimated_executed_block_flops_per_sequence"))
    ]
    if dense:
        dense_ce = min(float(row["val_loss"]) for row in dense)
        dense_flops = min(
            float(row["estimated_executed_block_flops_per_sequence"])
            for row in dense
        )
        dominated = sum(
            float(row["val_loss"]) >= dense_ce
            and float(row["estimated_executed_block_flops_per_sequence"]) >= dense_flops
            for row in grpo_rows
            if finite(row.get("val_loss"))
            and finite(row.get("estimated_executed_block_flops_per_sequence"))
        )
        if dominated == len(grpo_rows):
            criterion += (
                " Every GRPO point remained dominated by the matched dense model in the "
                "paper-style CE/FLOPs comparison."
            )
    return (
        method
        + (" Measured changes: " + "; ".join(details) + "." if details else "")
        + criterion
    )


def paper_report_markdown(rows: list[dict[str, Any]], experiments_dir: Path) -> str:
    table = paper_main_table(rows)
    seeds = {row.get("seed") for row in rows}
    return fr"""# SkipLayer Paper Reproduction on Tiny Shakespeare

## 1. Objective

This experiment first isolates the method in *Learning to Skip for Language Modeling*, then evaluates a router-only budget-guided GRPO extension without changing the Transformer weights. It asks whether a deeper, sparsely activated Transformer can match a shallower dense Transformer and whether RL fine-tuning moves that quality/compute point.

## 2. Scope and Scale Substitutions

{paper_setup_text(rows, experiments_dir)} The original private 1.6T-token corpus, 32K SentencePiece tokenizer, TPU grouped kernels, billion-parameter models, and 24-task one-shot evaluation suite are unavailable here. Results are therefore a method reproduction at small scale, not a reproduction of the paper's headline numbers.

## 3. Paper-Faithful SkipLayer

- Every physical Transformer layer has an independent bias-free linear `d_model → 2` router.
- The router consumes the pre-normalized layer input and emits skip/execute logits.
- Straight-through Gumbel-Softmax gives a hard binary forward path and soft backward surrogate.
- A skipped token is exact identity. An active token executes attention and FFN.
- Keys and values are retained for all causal-context tokens; only active queries and FFN inputs are gathered during greedy sparse evaluation.
- Greedy decoding selects the larger router logit.

## 4. Objective and Optimization

The paper's layerwise capacity objective is implemented as a sum, not an average:

$$L = L_{{\mathrm{{CE}}}} + 0.1\sum_l(r_l-P)^2.$$

There is no density-loss warmup. Training uses fixed-decay Adafactor with β1=0, β2=0.99, no separate gradient clipping, dropout 0, and inverse-square-root decay after 10,000 updates. A learning rate other than 0.1 is explicitly a scale-adapted ablation.

## 5. Main Results

{table}

With one seed, “± 0” means across-seed uncertainty is unavailable, not zero uncertainty.

## 6. Quality vs Compute

![Validation CE Pareto frontier](pareto_ce.png)

![Validation perplexity Pareto frontier](pareto_perplexity.png)

`compute_fraction` is router density relative to the sparse model's physical depth. It is not FLOPs relative to the shallower dense baseline. Paper-style estimated FLOPs include always-on key/value projections.

## 7. Routing Behavior

{routing_observations(rows)} Each sparse experiment additionally contains hard/soft routing heatmaps and a `token_skip_behavior.csv`/bubble plot analogous to the paper's token-skipping analysis.

## 8. Training Dynamics

Observed endpoint changes: {training_dynamics(experiments_dir)}

## 9. Computational Efficiency

Training keeps the dense candidate computation to preserve the straight-through gradient. Greedy evaluation uses real gathered active queries and FFN inputs while retaining all key/value projections. The Python gather/scatter implementation on MPS is a semantic reference and can be slower than dense execution; it is not comparable to the paper's specialized TPU grouped kernels.

## 10. Budget-Guided GRPO Extension

{paper_grpo_text(rows)}

![Supervised to GRPO movement](sl_vs_grpo.png)

## 11. Conclusion

{paper_conclusion(rows)}

## 12. Recommended Next Experiment

Repeat the matched supervised pair and the selected GRPO point for at least three seeds, then add 25% and 12.5% density rows while holding effective depth fixed.{'' if len(seeds) >= 3 else ' The present one-seed result is not a stable ranking.'}
"""


def mor_main_table(rows: list[dict[str, Any]]) -> str:
    dense = [row for row in rows if row.get("model") == "dense"]
    dense_flops, _ = mean_std(
        [row.get("estimated_executed_block_flops_per_sequence") for row in dense]
    )
    dense_params, _ = mean_std([row.get("parameter_count") for row in dense])
    headings = [
        "Model", "Seeds", "Validation CE", "Perplexity", "Layers/token",
        "Skipped", "Unique parameters", "Params vs dense", "Estimated block FLOPs",
        "FLOPs vs dense", "Generation tokens/s",
    ]
    lines = ["| " + " | ".join(headings) + " |", "|" + "---|" * len(headings)]
    for label, members in group_rows(rows):
        ce, ce_std = mean_std([row.get("val_loss") for row in members])
        ppl, ppl_std = mean_std([row.get("val_perplexity") for row in members])
        depth, depth_std = mean_std([row.get("layers_per_token") for row in members])
        skipped, skipped_std = mean_std([row.get("skip_fraction") for row in members])
        params, params_std = mean_std([row.get("parameter_count") for row in members])
        flops, flops_std = mean_std(
            [row.get("estimated_executed_block_flops_per_sequence") for row in members]
        )
        speed, speed_std = mean_std([row.get("generation_tokens_per_sec") for row in members])
        lines.append(
            "| " + " | ".join(
                [
                    label, str(len(members)), display(ce, ce_std), display(ppl, ppl_std),
                    display(depth, depth_std), display(skipped, skipped_std, True),
                    f"{params / 1e6:.3f}M ± {params_std / 1e6:.3f}M" if finite(params) else "NaN",
                    f"{params / dense_params:.3f}×" if finite(params) and finite(dense_params) else "NaN",
                    f"{flops / 1e6:.1f}M ± {flops_std / 1e6:.1f}M" if finite(flops) else "NaN",
                    f"{flops / dense_flops:.3f}×" if finite(flops) and finite(dense_flops) else "NaN",
                    display(speed, speed_std),
                ]
            ) + " |"
        )
    return "\n".join(lines)


def mor_conclusion(rows: list[dict[str, Any]]) -> str:
    complete = [
        row for row in rows
        if finite(row.get("val_loss"))
        and finite(row.get("estimated_executed_block_flops_per_sequence"))
    ]
    if not complete:
        return "The comparison is not complete."
    best_quality = min(complete, key=lambda row: float(row["val_loss"]))
    best_compute = min(
        complete,
        key=lambda row: float(row["estimated_executed_block_flops_per_sequence"]),
    )
    return (
        f"The lowest validation CE was produced by {model_label(best_quality)} "
        f"({float(best_quality['val_loss']):.4f}). The lowest estimated inference FLOPs "
        f"were produced by {model_label(best_compute)}. These are separate criteria: MoR "
        "reduces unique parameters through sharing, while routing reduces executed recursion "
        "work. A point improves the global frontier only if no other model has both lower CE "
        "and lower paper-style FLOPs."
    )


def mor_report_markdown(rows: list[dict[str, Any]], experiments_dir: Path) -> str:
    has_hybrid = any(row.get("model") == "mor_skip" for row in rows)
    objective = (
        "This matched Tiny Shakespeare experiment compares full dense, SkipLayer, "
        "SkipLayer + GRPO, MoR, native MoR recursion-routing GRPO, and the literal "
        "two-level MoR + SkipLayer hybrid before and after GRPO."
        if has_hybrid
        else "This matched Tiny Shakespeare experiment compares the five requested systems: "
        "a full eight-layer dense Transformer, supervised SkipLayer, SkipLayer + GRPO, "
        "Mixture-of-Recursions (MoR), and MoR + GRPO."
    )
    hybrid_text = (
        " The two-level hybrid keeps MoR admission fixed and adds six independent "
        "SkipLayer heads—one for each of the two shared blocks across three recursions. "
        "Its combined gate is `MoR_admitted × SkipLayer_execute`; entry and exit remain mandatory."
        if has_hybrid else ""
    )
    return fr"""# Mixture-of-Recursions + SkipLayer/GRPO Comparison

## 1. Objective

{objective}

## 2. MoR Architecture

The implementation follows the paper's selected expert-choice design at small scale: Middle-Cycle sharing with `1 + 2×3 + 1 = 8` effective layers, three recursions, capacities `1, 2/3, 1/3`, linear sigmoid routers, scale `α=0.1`, auxiliary BCE coefficient `0.001`, no capacity warmup, and recursion-wise attention/KV restriction. MoR therefore stores four unique Transformer blocks while exposing eight effective layers. Supervised training uses hierarchical top-k routing; greedy evaluation uses the learned `0.5` threshold and is reported separately from oracle top-k validation.{hybrid_text}

## 3. GRPO Extension

All GRPO variants freeze Transformer weights and update only the router under study. SkipLayer uses explicit layer budgets. Native MoR-GRPO uses recursion budgets. The two-level hybrid keeps the outer MoR router frozen and samples 25/50/75/100% conditional inner-execution budgets; its 100% execute-all-eligible path is the quality anchor excluded from policy loss. Every sampled action retains its exact behavior probability, and PPO ratios are computed only over valid token-routing decisions.

## 4. Main Results

{mor_main_table(rows)}

With one seed, `± 0` means across-seed uncertainty is unavailable.

## 5. Quality vs FLOPs

![Validation CE Pareto frontier](pareto_ce.png)

![Validation perplexity Pareto frontier](pareto_perplexity.png)

All FLOPs are forward-pass block estimates. MoR recursion-wise attention scales quadratically with the surviving token fraction because both queries and KV entries are restricted. Python/MPS wall-clock throughput is reported separately and is not an optimized-kernel claim.

## 6. Supervised to GRPO Movement

![Supervised to GRPO movement](sl_vs_grpo.png)

{grpo_assessment(rows)}

## 7. Routing Diagnostics

Each routed run contains layer heatmaps, token-skip analyses, difficulty/depth correlations, and per-layer utilization. MoR runs additionally log recursion utilization, soft router probabilities, auxiliary BCE, threshold accuracy, and greedy FLOPs. Hybrid runs separately log outer admission, conditional inner execution, combined block utilization, and every GRPO rollout budget.

## 8. Training Dynamics

{training_dynamics(experiments_dir)}

## 9. Conclusion

{mor_conclusion(rows)} This is a one-seed, character-level architecture study and not evidence that the ranking transfers directly to LLM scale.
"""


def report_markdown(rows: list[dict[str, Any]], experiments_dir: Path) -> str:
    if any(row.get("model") in {"mor", "mor_skip"} for row in rows):
        return mor_report_markdown(rows, experiments_dir)
    if rows and all(bool(row.get("paper_reproduction")) for row in rows):
        return paper_report_markdown(rows, experiments_dir)
    table, assessment = main_table(group_rows(rows)), grpo_assessment(rows)
    return fr"""# Sparse Transformer Routing Experiment

## 1. Objective

Sparse depth lets each token skip Transformer blocks it may not need. The linear SkipLayer baseline makes an independent skip/execute decision at each layer. The GRU router also remembers its earlier routing decisions across depth. GRPO is tested as a second-stage optimizer because final routing decisions are discrete and can be scored directly for quality and compute. H1 asks whether linear routing retains quality with less compute; H2 asks whether the GRU improves that tradeoff; H3 asks whether GRPO moves the Pareto frontier. H3 is not assumed true.

## 2. Experimental Setup

{setup_text(rows, experiments_dir)} Missing values are reported as `NaN`; no results are synthesized.

## 3. Models Compared

- **Dense Transformer:** every token executes every block.
- **Linear SkipLayer:** one two-logit linear gate per block, trained with straight-through Gumbel-Softmax.
- **GRU SkipLayer:** a per-token hidden state propagates across depth before producing each gate.
- **GRU SkipLayer + GRPO:** starts from the supervised GRU checkpoint and normally freezes the Transformer while updating only the router.

## 4. Supervised Routing Objective

$$L = L_{{\mathrm{{CE}}}} + \lambda\frac{{1}}{{L}}\sum_l(r_l-P)^2.$$

Here, $L_{{\mathrm{{CE}}}}$ is cross entropy, $r_l$ is hard executed-token density at layer $l$, $P$ is requested density, and $\lambda$ weights the density penalty. The penalty is warmed up rather than applied immediately.

## 5. GRPO Objective

The state $s_{{t,l}}$ contains the current token representation and GRU routing state. The action is skip or execute, and the GRU router is the policy:

$$R = -L_{{\mathrm{{CE}}}}-\lambda_c C-\beta\,KL.$$

The router is rewarded for maintaining prediction quality while using fewer layers. Advantages are normalized within each group, and trajectory log probability is the mean over token-layer decisions before applying the clipped GRPO objective.

## 6. Main Results

{table}

With one seed, “± 0” does not quantify uncertainty; it means across-seed deviation is unavailable.

## 7. Quality vs Compute

![Validation CE Pareto frontier](pareto_ce.png)

![Validation perplexity Pareto frontier](pareto_perplexity.png)

Points must be compared at matched compute. Lower CE with more layers is a tradeoff, not an unconditional improvement.

{fair_router_comparison(rows)}

## 8. Effect of GRPO

**{assessment}** This compares each GRPO run to its initializing supervised checkpoint. Exact equal-compute or equal-quality claims require overlapping sweep points.

![Supervised to GRPO movement](sl_vs_grpo.png)

## 9. Routing Behavior

{routing_observations(rows)} Routing heatmaps contain hard decisions and soft execute probabilities for the same validation text.

## 10. Training Dynamics

Each experiment's `plots/` directory contains CE, perplexity, accuracy, density, depth, and per-layer histories. GRPO runs additionally contain reward, component, KL, and entropy histories. Persistent near-zero entropy suggests policy collapse; no collapse is inferred when logs are absent.

Observed endpoint changes: {training_dynamics(experiments_dir)}

## 11. Computational Efficiency

`compute_fraction` is the theoretical fraction of token-layer block executions. Stage A still evaluates every candidate and masks its output, so it demonstrates logical sparsity but **does not provide sparse-kernel wall-clock acceleration**. Generation latency is measured separately.

## 12. Limitations

Tiny Shakespeare is extremely small and character-level modeling is simplistic. Routing overhead matters, ordinary kernels may not exploit token sparsity, results may not transfer to LLMs, GRPO adds substantial training compute, and small seed counts do not justify statistical-significance claims.

## 13. Conclusion

{conclusion_text(rows)}

## 14. Recommended Next Experiment

{recommendation_text(rows)}
"""


def image_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def markdown_to_html(markdown: str, results_dir: Path) -> str:
    """Render the report's Markdown subset without an online dependency."""
    out, paragraph, items, table = [], [], [], []

    def inline(value: str) -> str:
        escaped = html.escape(value)
        while "**" in escaped:
            escaped = escaped.replace("**", "<strong>", 1)
            if "**" not in escaped:
                break
            escaped = escaped.replace("**", "</strong>", 1)
        return escaped

    def flush() -> None:
        nonlocal paragraph, items, table
        if paragraph:
            out.append(f"<p>{inline(' '.join(paragraph))}</p>")
            paragraph = []
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(item)}</li>" for item in items) + "</ul>")
            items = []
        if table:
            rows = [row for row in table if not all(set(cell) <= {"-", ":"} for cell in row)]
            out.append("<table><thead><tr>" + "".join(f"<th>{inline(cell)}</th>" for cell in rows[0]) + "</tr></thead><tbody>")
            out.extend("<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row) + "</tr>" for row in rows[1:])
            out.append("</tbody></table>")
            table = []

    for line in markdown.splitlines():
        value = line.strip()
        if value.startswith("|") and value.endswith("|"):
            if paragraph or items:
                flush()
            table.append([cell.strip() for cell in value.strip("|").split("|")])
        elif value.startswith("#"):
            flush()
            level = min(len(value) - len(value.lstrip("#")), 6)
            out.append(f"<h{level}>{inline(value[level:].strip())}</h{level}>")
        elif value.startswith("- "):
            if paragraph or table:
                flush()
            items.append(value[2:])
        elif value.startswith("![") and "](" in value and value.endswith(")"):
            flush()
            alt, filename = value[2:].split("](", 1)
            uri = image_uri(results_dir / filename[:-1])
            out.append(f'<img alt="{html.escape(alt)}" src="{uri}" />' if uri else f'<p class="missing">Plot unavailable: {html.escape(alt)}</p>')
        elif value.startswith("$$") and value.endswith("$$"):
            flush()
            out.append(f"<pre class=equation>{html.escape(value[2:-2])}</pre>")
        elif not value:
            flush()
        else:
            if items or table:
                flush()
            paragraph.append(value)
    flush()
    body = "\n".join(out)
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>Sparse Transformer Routing Experiment</title><style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#17202a}}h1,h2{{color:#123b5d}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ccd6dd;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%;height:auto;margin:12px 0}}pre.equation{{background:#f3f6f8;padding:14px;overflow:auto}}.missing{{color:#777;font-style:italic}}</style></head><body>{body}</body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    experiments_dir, results_dir = Path(args.experiments_dir), Path(args.results_dir)
    rows = []
    for path in sorted(experiments_dir.glob("*/summary.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            pass
    if not rows:
        raise SystemExit("No completed summary.json files found; refusing to fabricate a report")
    results_dir.mkdir(parents=True, exist_ok=True)
    markdown = report_markdown(rows, experiments_dir)
    (results_dir / "REPORT.md").write_text(markdown, encoding="utf-8")
    (results_dir / "REPORT.html").write_text(markdown_to_html(markdown, results_dir), encoding="utf-8")
    print(f"wrote {results_dir / 'REPORT.md'} and {results_dir / 'REPORT.html'}")


if __name__ == "__main__":
    main()
