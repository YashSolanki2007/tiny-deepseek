"""Evaluate one checkpoint or every completed experiment and add analysis artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from analysis import (
    difficulty_depth_analysis,
    save_difficulty_plot,
    save_token_skip_artifacts,
    token_skip_analysis,
)
from data import TinyShakespeareData
from evaluation import evaluate_model
from generate import generate_tokens
from utils import (
    estimate_dense_block_flops,
    estimate_mor_flops,
    estimate_skiplayer_flops,
    load_checkpoint,
    select_device,
    write_json,
)
from visualize_routing import save_routing_heatmap


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_path: str,
    device: torch.device,
    batch_size: int,
    eval_iters: int,
    generation_tokens: int,
) -> dict:
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    dataset = TinyShakespeareData(data_path, model.config.context_length)
    if dataset.stoi != checkpoint["stoi"]:
        raise ValueError(f"Vocabulary mismatch for {checkpoint_path}")
    training = checkpoint.get("training_config") or {}
    target = float(
        training.get("target_density", training.get("training", {}).get("target_density", 1.0))
    )
    metrics = evaluate_model(model, dataset, device, batch_size, eval_iters, target)
    if model.config.paper_reproduction:
        if model.config.model_type == "dense":
            estimated_flops = estimate_dense_block_flops(
                model.config, model.config.context_length
            )
        elif model.config.model_type == "mor":
            estimated_flops = estimate_mor_flops(
                model.config,
                model.config.context_length,
                metrics["recursion_utilization"],
            )
        else:
            estimated_flops = estimate_skiplayer_flops(
                model.config,
                model.config.context_length,
                metrics["compute_fraction"],
            )
        metrics["estimated_executed_block_flops_per_sequence"] = estimated_flops
        metrics["estimated_flops_vs_full_dense"] = estimated_flops / estimate_dense_block_flops(
            model.config, model.config.context_length
        )
    prompt = "ROMEO:"
    ids = torch.tensor([[dataset.stoi[ch] for ch in prompt]], dtype=torch.long, device=device)
    generated_ids, generation_gates, elapsed = generate_tokens(
        model, ids, generation_tokens, temperature=0.0, top_k=None
    )
    metrics["generation_tokens_per_sec"] = generation_tokens / max(elapsed, 1e-9)
    metrics["generation_ms_per_token"] = 1000 * elapsed / max(generation_tokens, 1)
    if generation_gates is not None:
        metrics["generation_compute_fraction"] = generation_gates.mean().item()
    difficulty = difficulty_depth_analysis(model, dataset, device, batch_size, eval_iters)
    metrics.update(difficulty)
    experiment_dir = checkpoint_path.parent.parent
    samples_dir = experiment_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample = dataset.decode(generated_ids[0].detach().cpu().tolist())
    (samples_dir / "evaluation_sample.txt").write_text(sample + "\n", encoding="utf-8")
    plots = experiment_dir / "plots"
    save_difficulty_plot(difficulty, plots / "difficulty_vs_depth")
    if model.config.model_type in {"sparse", "mor"}:
        routing_dir = experiment_dir / "routing_visualizations"
        text = "ROMEO:\nWhat light through yonder window breaks?"
        save_routing_heatmap(model, dataset.stoi, text, "soft", routing_dir / "routing_soft.png")
        save_routing_heatmap(model, dataset.stoi, text, "hard", routing_dir / "routing_hard.png")
        token_skip = token_skip_analysis(model, dataset, device, batch_size, eval_iters)
        save_token_skip_artifacts(token_skip, plots / "token_skip_behavior")
        metrics["most_skipped_tokens"] = token_skip["most_skipped_tokens"]
        metrics["least_skipped_tokens"] = token_skip["least_skipped_tokens"]
    summary_path = experiment_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update(metrics)
    if model.config.model_type == "sparse":
        block_parameters = sum(
            parameter.numel()
            for block in model.blocks
            for parameter in block.parameters()
        )
        router_parameters = sum(
            parameter.numel() for parameter in model.router.parameters()
        )
        always_active = model.parameter_count() - block_parameters - router_parameters
        summary["active_parameter_estimate"] = (
            always_active
            + router_parameters
            + metrics["compute_fraction"] * block_parameters
        )
    elif model.config.model_type == "mor":
        summary["active_parameter_estimate"] = model.parameter_count()
    summary["checkpoint"] = str(checkpoint_path)
    write_json(summary_path, summary)
    print(
        f"{experiment_dir.name}: val_ce={metrics['val_loss']:.4f} "
        f"ppl={metrics['val_perplexity']:.3f} depth={metrics['layers_per_token']:.2f}/"
        f"{model.config.n_layers} generation={metrics['generation_tokens_per_sec']:.2f} tok/s"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--data-path", default="data/input.txt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--generation-tokens", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not args.checkpoint and not args.all:
        raise SystemExit("Provide --checkpoint or --all")
    device = select_device(args.device)
    if args.checkpoint:
        checkpoints = [Path(args.checkpoint)]
    else:
        checkpoints = []
        for summary_path in sorted(Path(args.experiments_dir).glob("*/summary.json")):
            summary = json.loads(summary_path.read_text())
            candidate = Path(summary.get("checkpoint", ""))
            if candidate.exists():
                checkpoints.append(candidate)
            elif (summary_path.parent / "checkpoints" / "best_val_loss.pt").exists():
                checkpoints.append(summary_path.parent / "checkpoints" / "best_val_loss.pt")
    if not checkpoints:
        raise SystemExit("No checkpoints found")
    for checkpoint in checkpoints:
        evaluate_checkpoint(
            checkpoint, args.data_path, device, args.batch_size,
            args.eval_iters, args.generation_tokens,
        )


if __name__ == "__main__":
    main()
