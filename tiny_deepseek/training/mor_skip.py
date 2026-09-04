"""Supervised inner-SkipLayer training on a frozen paper-style MoR model."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

import torch

from tiny_deepseek.core.config import ModelConfig
from tiny_deepseek.data.shakespeare import TinyShakespeareData
from tiny_deepseek.evaluation.metrics import evaluate_model
from tiny_deepseek.training.logging import StructuredLogger
from tiny_deepseek.core.losses import masked_density_loss
from tiny_deepseek.core.model import MixtureOfRecursionsSkipLayerTransformer, build_model
from tiny_deepseek.core.utils import (
    estimate_dense_block_flops,
    estimate_mor_skip_flops,
    gradient_norm,
    load_checkpoint,
    restore_training_state,
    routing_metrics,
    save_checkpoint,
    select_device,
    set_seed,
    synchronize_device,
    top1_accuracy,
    write_json,
)


BASE_FIELDS = [
    "step", "split", "train_loss", "train_accuracy", "total_loss",
    "skip_density_loss", "lambda_skip_density", "target_skip_density",
    "val_loss", "val_perplexity", "val_accuracy", "layers_per_token",
    "compute_fraction", "skip_fraction", "routing_entropy",
    "mean_conditional_skip_density", "mean_recursions_per_token",
    "mor_aux_loss", "mor_router_accuracy", "estimated_block_flops",
    "estimated_flops_vs_full_dense", "learning_rate", "gradient_norm",
    "skip_router_gradient_norm", "frozen_parameter_gradient_norm",
    "tokens_processed", "seconds_per_step", "tokens_per_second",
    "validation_time_sec",
]


def add_hybrid_flops(
    metrics: dict[str, Any], model: MixtureOfRecursionsSkipLayerTransformer
) -> dict[str, Any]:
    executed = estimate_mor_skip_flops(
        model.config,
        model.config.context_length,
        metrics["recursion_utilization"],
        metrics["combined_block_utilization"],
    )
    dense = estimate_dense_block_flops(model.config, model.config.context_length)
    return {
        **metrics,
        "estimated_executed_block_flops_per_sequence": executed,
        "estimated_block_flops": executed,
        "estimated_flops_vs_full_dense": executed / dense,
    }


def upgrade_mor_checkpoint(
    checkpoint_path: str, device: torch.device, initial_execute_probability: float
) -> tuple[MixtureOfRecursionsSkipLayerTransformer, dict[str, Any]]:
    source, checkpoint = load_checkpoint(checkpoint_path, device)
    if source.config.model_type != "mor":
        raise ValueError("The hybrid must start from a supervised MoR checkpoint")
    values = source.config.to_dict()
    values.update(
        model_type="mor_skip",
        initial_execute_probability=initial_execute_probability,
        router_bias=True,
        sparse_inference=True,
    )
    model = build_model(ModelConfig.from_dict(values)).to(device)
    missing, unexpected = model.load_state_dict(source.state_dict(), strict=False)
    if unexpected or any(not key.startswith("skip_router.") for key in missing):
        raise RuntimeError(
            f"Unexpected MoR-to-hybrid state mismatch; missing={missing}, unexpected={unexpected}"
        )
    return model, checkpoint


def metric_fields(model: MixtureOfRecursionsSkipLayerTransformer) -> list[str]:
    fields = list(BASE_FIELDS)
    for recursion in range(model.config.recursion_steps):
        fields += [
            f"mor_recursion_{recursion + 1}_admission",
            f"mor_recursion_{recursion + 1}_soft_admission",
        ]
        for block in range(model.config.recursion_block_layers):
            label = f"r{recursion + 1}_b{block + 1}"
            fields += [
                f"skip_{label}_conditional_execute",
                f"skip_{label}_soft_conditional_execute",
                f"combined_{label}_utilization",
            ]
    return fields


def add_router_fields(row: dict[str, Any], metrics: dict[str, Any], model) -> None:
    for index, value in enumerate(metrics["recursion_utilization"]):
        row[f"mor_recursion_{index + 1}_admission"] = value
    for index, value in enumerate(metrics["recursion_soft_utilization"]):
        row[f"mor_recursion_{index + 1}_soft_admission"] = value
    inner = 0
    for recursion in range(model.config.recursion_steps):
        for block in range(model.config.recursion_block_layers):
            label = f"r{recursion + 1}_b{block + 1}"
            row[f"skip_{label}_conditional_execute"] = metrics[
                "skip_conditional_utilization"
            ][inner]
            row[f"skip_{label}_soft_conditional_execute"] = metrics[
                "skip_soft_conditional_utilization"
            ][inner]
            row[f"combined_{label}_utilization"] = metrics[
                "combined_block_utilization"
            ][inner]
            inner += 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", help="Completed supervised MoR checkpoint")
    result.add_argument("--resume", help="Resume a supervised MoR+Skip checkpoint")
    result.add_argument("--experiment-dir")
    result.add_argument("--data-path", default="data/input.txt")
    result.add_argument("--device", default="auto")
    result.add_argument("--max-steps", type=int, default=1000)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--eval-interval", type=int, default=100)
    result.add_argument("--eval-iters", type=int, default=20)
    result.add_argument("--log-interval", type=int, default=20)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--target-skip-density", type=float, default=0.5)
    result.add_argument("--lambda-skip-density", type=float, default=0.1)
    result.add_argument("--initial-execute-probability", type=float, default=0.9)
    result.add_argument("--grad-clip", type=float, default=1.0)
    result.add_argument("--seed", type=int, default=42)
    return result


def main() -> None:
    args = parser().parse_args()
    if not args.checkpoint and not args.resume:
        raise SystemExit("Provide --checkpoint for a new hybrid or --resume")
    if not 0 <= args.target_skip_density <= 1:
        raise ValueError("target-skip-density must lie in [0,1]")
    device = select_device(args.device)
    set_seed(args.seed)
    resume_checkpoint = None
    if args.resume:
        model, resume_checkpoint = load_checkpoint(args.resume, device)
        if not isinstance(model, MixtureOfRecursionsSkipLayerTransformer):
            raise ValueError("Resume checkpoint is not a MoR+Skip model")
        source_checkpoint = resume_checkpoint
    else:
        model, source_checkpoint = upgrade_mor_checkpoint(
            args.checkpoint, device, args.initial_execute_probability
        )
    dataset = TinyShakespeareData(args.data_path, model.config.context_length)
    if dataset.stoi != source_checkpoint["stoi"]:
        raise ValueError("Checkpoint vocabulary differs from the dataset")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    skip_parameters = list(model.skip_router_parameters())
    for parameter in skip_parameters:
        parameter.requires_grad_(True)
    frozen_parameters = [p for p in model.parameters() if not p.requires_grad]
    optimizer = torch.optim.AdamW(skip_parameters, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.max_steps, 1), eta_min=args.learning_rate * 0.1
    )
    start_step = 0
    best = {"val_loss": float("inf"), "quality_compute_score": float("inf")}
    if resume_checkpoint:
        start_step = restore_training_state(resume_checkpoint, optimizer, scheduler)
        best.update(resume_checkpoint.get("best_metrics") or {})

    source_dir = Path(args.checkpoint).parent.parent if args.checkpoint else Path(args.resume).parent.parent
    experiment_dir = Path(
        args.experiment_dir
        or f"artifacts/experiments/{source_dir.name}_inner_skip_P{int(100*args.target_skip_density):03d}_seed{args.seed}"
    )
    for child in ("checkpoints", "plots", "samples", "routing_visualizations"):
        (experiment_dir / child).mkdir(parents=True, exist_ok=True)
    full_config = {
        "stage": "mor_inner_skip_supervised",
        "model": model.config.to_dict(),
        "source_mor_checkpoint": args.checkpoint,
        "training": vars(args),
        "target_density": args.target_skip_density,
        "frozen_backbone": True,
        "frozen_mor_router": True,
    }
    write_json(experiment_dir / "config.json", full_config)
    logger = StructuredLogger(
        experiment_dir,
        metric_fields(model),
        purge_step=start_step if resume_checkpoint else None,
    )

    latest_eval = add_hybrid_flops(
        evaluate_model(model, dataset, device, args.batch_size, args.eval_iters), model
    )
    latest_train: dict[str, Any] = {}
    tokens_processed = start_step * args.batch_size * model.config.context_length
    measured_seconds = 0.0
    training_started = time.perf_counter()
    try:
        for step in range(start_step, args.max_steps):
            model.train()
            synchronize_device(device)
            started = time.perf_counter()
            x, y = dataset.get_batch("train", args.batch_size, device)
            output = model(x, y, routing_mode="topk")
            penalty = masked_density_loss(
                output.skip_hard_gates,
                output.routing_decision_mask,
                args.target_skip_density,
            )
            total = output.lm_loss + args.lambda_skip_density * penalty
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            skip_grad = gradient_norm(skip_parameters)
            frozen_grad = gradient_norm(frozen_parameters)
            torch.nn.utils.clip_grad_norm_(skip_parameters, args.grad_clip)
            optimizer.step()
            scheduler.step()
            synchronize_device(device)
            seconds = time.perf_counter() - started
            measured_seconds += seconds
            tokens_processed += x.numel()
            metrics = add_hybrid_flops(
                routing_metrics(output, model.config.n_layers), model
            )
            latest_train = {
                "split": "train",
                "train_loss": output.lm_loss.item(),
                "train_accuracy": top1_accuracy(output.logits, y).item(),
                "total_loss": total.item(),
                "skip_density_loss": penalty.item(),
                "lambda_skip_density": args.lambda_skip_density,
                "target_skip_density": args.target_skip_density,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": skip_grad,
                "skip_router_gradient_norm": skip_grad,
                "frozen_parameter_gradient_norm": frozen_grad,
                "tokens_processed": tokens_processed,
                "seconds_per_step": seconds,
                "tokens_per_second": x.numel() / max(seconds, 1e-9),
                **metrics,
            }
            add_router_fields(latest_train, metrics, model)
            if step % args.log_interval == 0 or step == args.max_steps - 1:
                logger.log(latest_train, step)
                print(
                    f"step {step:5d} | ce {output.lm_loss.item():.4f} | "
                    f"conditional skip density {metrics['mean_conditional_skip_density']:.3f} | "
                    f"depth {metrics['layers_per_token']:.2f}/8 | "
                    f"FLOPs {metrics['estimated_flops_vs_full_dense']:.3f}x dense"
                )
            if (step + 1) % args.eval_interval == 0 or step == args.max_steps - 1:
                latest_eval = add_hybrid_flops(
                    evaluate_model(
                        model, dataset, device, args.batch_size, args.eval_iters,
                        routing_mode="greedy",
                    ),
                    model,
                )
                row = {"split": "validation", **latest_eval}
                add_router_fields(row, latest_eval, model)
                logger.log(row, step)
                score = latest_eval["val_loss"] + latest_eval["estimated_flops_vs_full_dense"]
                checkpoint_kwargs = dict(
                    model=model, optimizer=optimizer, scheduler=scheduler, step=step + 1,
                    stoi=dataset.stoi, itos=dataset.itos, training_config=full_config,
                    best_metrics=best,
                )
                if latest_eval["val_loss"] < best["val_loss"]:
                    best["val_loss"] = latest_eval["val_loss"]
                    save_checkpoint(experiment_dir / "checkpoints/best_val_loss.pt", **checkpoint_kwargs)
                if score < best["quality_compute_score"]:
                    best["quality_compute_score"] = score
                    save_checkpoint(experiment_dir / "checkpoints/best_quality_compute.pt", **checkpoint_kwargs)
                save_checkpoint(experiment_dir / "checkpoints/latest.pt", **checkpoint_kwargs)
                print(
                    f"validation | ce {latest_eval['val_loss']:.4f} | "
                    f"accuracy {latest_eval['val_accuracy']:.3f} | "
                    f"depth {latest_eval['layers_per_token']:.2f}/8 | "
                    f"FLOPs {latest_eval['estimated_flops_vs_full_dense']:.3f}x dense"
                )
    finally:
        logger.close()

    training_seconds = time.perf_counter() - training_started
    selected = experiment_dir / "checkpoints/best_quality_compute.pt"
    summary = {
        "model": "mor_skip",
        "router_type": "mor_expert_plus_inner_skiplayer",
        "training_method": "supervised",
        "experiment_family": "mor_skip_hybrid",
        "paper_reproduction": True,
        "mor_reproduction": True,
        "seed": args.seed,
        "target_density": args.target_skip_density,
        "lambda_density": args.lambda_skip_density,
        "parameter_count": model.parameter_count(),
        "active_parameter_estimate": model.parameter_count(),
        "training_time_sec": training_seconds,
        "seconds_per_step": measured_seconds / max(args.max_steps - start_step, 1),
        "tokens_per_sec": (
            (args.max_steps - start_step) * args.batch_size * model.config.context_length
            / max(training_seconds, 1e-9)
        ),
        "checkpoint": str(selected),
        "source_mor_checkpoint": args.checkpoint,
        **latest_eval,
    }
    write_json(experiment_dir / "summary.json", summary)
    save_checkpoint(
        path=experiment_dir / "checkpoints/latest.pt", model=model,
        optimizer=optimizer, scheduler=scheduler, step=args.max_steps,
        stoi=dataset.stoi, itos=dataset.itos, training_config=full_config,
        best_metrics=best, summary=summary,
    )
    print(f"completed supervised MoR+Skip experiment: {experiment_dir}")


if __name__ == "__main__":
    main()
