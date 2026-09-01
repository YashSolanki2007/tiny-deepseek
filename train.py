"""Supervised training for dense, linear SkipLayer, and GRU SkipLayer models."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict

import torch

from config import ModelConfig, TrainConfig
from data import TinyShakespeareData
from evaluation import evaluate_model
from logging_utils import StructuredLogger
from losses import density_loss, scheduled_coefficient
from model import TransformerBase, build_model
from optimizers import FixedDecayAdafactor
from utils import (
    estimate_dense_block_flops,
    estimate_mor_flops,
    estimate_skiplayer_flops,
    gradient_norm,
    load_checkpoint,
    perplexity,
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
    "step", "split", "train_loss", "train_perplexity", "train_accuracy",
    "val_loss", "val_perplexity", "val_accuracy", "total_loss", "density_loss",
    "lambda_density", "target_density", "mean_soft_gate", "mean_hard_gate",
    "layers_per_token", "compute_fraction", "skip_fraction", "routing_entropy",
    "learning_rate", "gradient_norm", "router_gradient_norm", "transformer_gradient_norm",
    "tokens_processed", "seconds_per_step", "tokens_per_second", "validation_time_sec",
    "mor_aux_loss", "mor_router_accuracy", "mean_recursions_per_token",
    "estimated_block_flops", "estimated_flops_vs_full_dense",
]


def lr_factor(step: int, cfg: TrainConfig) -> float:
    if cfg.lr_schedule == "paper_inverse_sqrt":
        update = step + 1
        if update <= cfg.constant_lr_steps:
            return 1.0
        return math.sqrt(cfg.constant_lr_steps / update)
    minimum = cfg.min_lr / cfg.learning_rate
    if step < cfg.warmup_steps:
        return max((step + 1) / max(cfg.warmup_steps, 1), minimum)
    ratio = min(max((step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1), 0), 1)
    return minimum + 0.5 * (1 + math.cos(math.pi * ratio)) * (1 - minimum)


def active_parameter_estimate(model: TransformerBase, density: float) -> float:
    block_parameters = sum(p.numel() for block in model.blocks for p in block.parameters())
    router_parameters = sum(p.numel() for p in model.router_parameters())
    always_active = model.parameter_count() - block_parameters - router_parameters
    return float(always_active + router_parameters + density * block_parameters)


def execution_flop_metrics(
    model: TransformerBase, metrics: Dict[str, Any]
) -> tuple[float, float]:
    dense = estimate_dense_block_flops(model.config, model.config.context_length)
    if model.config.model_type == "mor":
        executed = estimate_mor_flops(
            model.config,
            model.config.context_length,
            metrics["recursion_utilization"],
        )
    elif model.config.model_type == "sparse" and model.config.paper_reproduction:
        executed = estimate_skiplayer_flops(
            model.config, model.config.context_length, metrics["compute_fraction"]
        )
    else:
        executed = dense * metrics["compute_fraction"]
    return executed, executed / dense


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["dense", "sparse", "dynamic", "mor"], default="sparse")
    parser.add_argument("--router", choices=["linear", "gru"], default="gru")
    parser.add_argument("--data-path", default="data/input.txt")
    parser.add_argument("--experiment-dir")
    parser.add_argument("--resume")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--router-dim", type=int, default=32)
    parser.add_argument("--gumbel-temperature", type=float, default=1.0)
    parser.add_argument("--initial-execute-probability", type=float, default=0.9)
    parser.add_argument("--no-tie-weights", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--target-density", type=float, default=0.5)
    parser.add_argument("--lambda-density", type=float, default=0.1)
    parser.add_argument("--density-warmup-start", type=float, default=0.10)
    parser.add_argument("--density-warmup-end", type=float, default=0.30)
    parser.add_argument("--quality-compute-alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--paper-reproduction",
        action="store_true",
        help="Use the architecture, objective, and optimizer reported by SkipLayer, scaled to this dataset.",
    )
    parser.add_argument("--optimizer", choices=["adamw", "adafactor"], default="adamw")
    parser.add_argument(
        "--lr-schedule", choices=["cosine", "paper_inverse_sqrt"], default="cosine"
    )
    parser.add_argument("--constant-lr-steps", type=int, default=10_000)
    parser.add_argument("--density-reduction", choices=["mean", "sum"], default="mean")
    parser.add_argument(
        "--paper-learning-rate",
        type=float,
        help="Optional scale-adapted Adafactor LR; omitted means the paper's exact 0.1.",
    )
    parser.add_argument("--recursion-steps", type=int, default=3)
    parser.add_argument("--recursion-block-layers", type=int, default=2)
    parser.add_argument(
        "--mor-capacity-factors", type=float, nargs="+", default=[1.0, 2 / 3, 1 / 3]
    )
    parser.add_argument("--mor-router-alpha", type=float, default=0.1)
    parser.add_argument("--mor-aux-loss-coefficient", type=float, default=0.001)
    parser.add_argument("--mor-capacity-warmup-steps", type=int, default=0)
    return parser


def save_named_checkpoint(
    path: Path,
    model: TransformerBase,
    optimizer,
    scheduler,
    step: int,
    dataset: TinyShakespeareData,
    config: Dict[str, Any],
    best: Dict[str, Any],
    summary: Dict[str, Any] | None = None,
) -> None:
    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        stoi=dataset.stoi,
        itos=dataset.itos,
        training_config=config,
        best_metrics=best,
        summary=summary,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.paper_reproduction:
        if args.model not in {"dense", "sparse", "mor"}:
            raise ValueError("The scaled protocol supports dense, sparse, or MoR models")
        if args.model == "sparse" and args.router != "linear":
            raise ValueError("The SkipLayer paper uses the linear two-class router")
        # Exact reported method/training choices that remain meaningful after
        # scaling the private 1.6T-token experiment to Tiny Shakespeare.
        args.dropout = 0.0
        args.d_ff = 8 * args.d_model
        args.initial_execute_probability = 0.5
        args.lambda_density = 0.1
        args.density_warmup_start = 0.0
        args.density_warmup_end = 0.0
        args.optimizer = "adafactor"
        args.learning_rate = (
            args.paper_learning_rate if args.paper_learning_rate is not None else 0.1
        )
        args.weight_decay = 0.0
        args.grad_clip = 0.0
        args.lr_schedule = "paper_inverse_sqrt"
        args.constant_lr_steps = 10_000
        args.density_reduction = "sum"
    device = select_device(args.device)
    set_seed(args.seed, args.deterministic)
    resume_checkpoint = None
    if args.resume:
        model, resume_checkpoint = load_checkpoint(args.resume, device)
        dataset = TinyShakespeareData(args.data_path, model.config.context_length)
        if dataset.stoi != resume_checkpoint["stoi"]:
            raise ValueError("Resume checkpoint vocabulary differs from dataset")
    else:
        dataset = TinyShakespeareData(args.data_path, args.context_length)
        model_type = "sparse" if args.model == "dynamic" else args.model
        model = build_model(
            ModelConfig(
                vocab_size=dataset.vocab_size,
                context_length=args.context_length,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                d_ff=args.d_ff,
                dropout=args.dropout,
                model_type=model_type,
                router_type=args.router,
                router_dim=args.router_dim,
                gumbel_temperature=args.gumbel_temperature,
                initial_execute_probability=args.initial_execute_probability,
                router_input_norm=args.paper_reproduction,
                router_bias=not args.paper_reproduction,
                paper_reproduction=args.paper_reproduction,
                sparse_inference=args.paper_reproduction,
                tie_weights=not args.no_tie_weights,
                mor_reproduction=model_type == "mor",
                recursion_steps=args.recursion_steps,
                recursion_block_layers=args.recursion_block_layers,
                mor_capacity_factors=tuple(args.mor_capacity_factors),
                mor_router_alpha=args.mor_router_alpha,
                mor_aux_loss_coefficient=args.mor_aux_loss_coefficient,
                mor_capacity_warmup_steps=args.mor_capacity_warmup_steps,
            )
        ).to(device)

    cfg = TrainConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        log_interval=args.log_interval,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        lambda_density=args.lambda_density,
        target_density=args.target_density,
        density_warmup_start=args.density_warmup_start,
        density_warmup_end=args.density_warmup_end,
        quality_compute_alpha=args.quality_compute_alpha,
        optimizer_name=args.optimizer,
        lr_schedule=args.lr_schedule,
        constant_lr_steps=args.constant_lr_steps,
        density_reduction=args.density_reduction,
        seed=args.seed,
    )
    density_label = f"P{int(round(cfg.target_density * 100)):03d}"
    default_name = (
        f"dense_seed{cfg.seed}" if model.config.model_type == "dense"
        else f"mor_R{model.config.recursion_steps}_seed{cfg.seed}"
        if model.config.model_type == "mor"
        else f"{model.config.router_type}_{density_label}_seed{cfg.seed}"
    )
    experiment_dir = Path(args.experiment_dir or f"experiments/{default_name}")
    for child in ("checkpoints", "plots", "samples", "routing_visualizations"):
        (experiment_dir / child).mkdir(parents=True, exist_ok=True)
    full_config = {
        "stage": "supervised", "model": model.config.to_dict(), "training": cfg.to_dict(),
        "device": str(device), "experiment_dir": str(experiment_dir),
    }
    write_json(experiment_dir / "config.json", full_config)

    if cfg.optimizer_name == "adafactor":
        optimizer = FixedDecayAdafactor(
            model.parameters(), lr=cfg.learning_rate, beta2=0.99,
            weight_decay=cfg.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, cfg))
    best = {
        "val_loss": float("inf"), "val_perplexity": float("inf"),
        "quality_compute_score": float("inf"),
    }
    start_step = 0
    if resume_checkpoint:
        start_step = restore_training_state(resume_checkpoint, optimizer, scheduler)
        best.update(resume_checkpoint.get("best_metrics") or {})

    fields = BASE_FIELDS + [f"layer_{i}_density" for i in range(model.config.n_layers)]
    fields += [f"layer_{i}_soft_probability" for i in range(model.config.n_layers)]
    if model.config.model_type == "mor":
        fields += [
            f"recursion_{i + 1}_{kind}"
            for i in range(model.config.recursion_steps)
            for kind in ("utilization", "soft_utilization")
        ]
    logger = StructuredLogger(
        experiment_dir, fields, purge_step=start_step if resume_checkpoint else None
    )
    router_params = list(model.router_parameters())
    router_ids = {id(p) for p in router_params}
    transformer_params = [p for p in model.parameters() if id(p) not in router_ids]
    tokens_processed = start_step * cfg.batch_size * model.config.context_length
    measured_step_seconds = 0.0
    training_started = time.perf_counter()
    last_eval: Dict[str, Any] = {}
    if model.config.model_type == "sparse":
        effective_target_density = cfg.target_density
    elif model.config.model_type == "mor":
        expected_depth = 2 + model.config.recursion_block_layers * sum(
            model.config.mor_capacity_factors
        )
        effective_target_density = expected_depth / model.config.n_layers
    else:
        effective_target_density = 1.0
    print(
        f"device={device} model={model.config.model_type} "
        f"router={model.config.router_type if model.config.model_type == 'sparse' else 'expert_linear' if model.config.model_type == 'mor' else 'none'} "
        f"parameters={model.parameter_count():,} experiment={experiment_dir}"
    )
    try:
        for step in range(start_step, cfg.max_steps):
            model.train()
            coefficient = (
                scheduled_coefficient(
                    step, cfg.max_steps, cfg.lambda_density,
                    cfg.density_warmup_start, cfg.density_warmup_end,
                ) if model.config.model_type == "sparse" else 0.0
            )
            synchronize_device(device)
            step_started = time.perf_counter()
            x, y = dataset.get_batch("train", cfg.batch_size, device)
            output = model(
                x,
                y,
                routing_mode="topk" if model.config.model_type == "mor" else "gumbel",
            )
            penalty = (
                density_loss(
                    output.hard_gates, cfg.target_density,
                    reduction=cfg.density_reduction,
                )
                if model.config.model_type == "sparse" else output.lm_loss.new_zeros(())
            )
            auxiliary = (
                output.mor_aux_loss
                if output.mor_aux_loss is not None
                else output.lm_loss.new_zeros(())
            )
            total = (
                output.lm_loss
                + coefficient * penalty
                + model.config.mor_aux_loss_coefficient * auxiliary
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            router_grad = gradient_norm(router_params)
            transformer_grad = gradient_norm(transformer_params)
            total_grad = gradient_norm(model.parameters())
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            synchronize_device(device)
            step_seconds = time.perf_counter() - step_started
            measured_step_seconds += step_seconds
            tokens_processed += x.numel()
            route = routing_metrics(output, model.config.n_layers)
            train_flops, train_flops_relative = execution_flop_metrics(model, route)
            if step % cfg.log_interval == 0 or step == cfg.max_steps - 1:
                row: Dict[str, Any] = {
                    "split": "train", "train_loss": output.lm_loss.item(),
                    "train_perplexity": perplexity(output.lm_loss.item()),
                    "train_accuracy": top1_accuracy(output.logits, y).item(),
                    "total_loss": total.item(), "density_loss": penalty.item(),
                    "lambda_density": coefficient, "target_density": effective_target_density,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": total_grad, "router_gradient_norm": router_grad,
                    "transformer_gradient_norm": transformer_grad,
                    "tokens_processed": tokens_processed, "seconds_per_step": step_seconds,
                    "tokens_per_second": x.numel() / max(step_seconds, 1e-9), **route,
                    "mor_aux_loss": auxiliary.item(),
                    "estimated_block_flops": train_flops,
                    "estimated_flops_vs_full_dense": train_flops_relative,
                }
                row.update({f"layer_{i}_density": v for i, v in enumerate(route["layer_utilization"])})
                row.update({f"layer_{i}_soft_probability": v for i, v in enumerate(route["layer_soft_probability"])})
                if model.config.model_type == "mor":
                    row.update(
                        {
                            f"recursion_{i + 1}_utilization": value
                            for i, value in enumerate(route["recursion_utilization"])
                        }
                    )
                    row.update(
                        {
                            f"recursion_{i + 1}_soft_utilization": value
                            for i, value in enumerate(route["recursion_soft_utilization"])
                        }
                    )
                logger.log(row, step)
                print(
                    f"step {step:5d} | ce {row['train_loss']:.4f} | ppl {row['train_perplexity']:.3f} "
                    f"| acc {row['train_accuracy']:.3f} | density {route['compute_fraction']:.3f} "
                    f"| depth {route['layers_per_token']:.2f}/{model.config.n_layers} "
                    f"| density_loss {penalty.item():.4f} | lambda {coefficient:.3f}"
                )

            if (step + 1) % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
                last_eval = evaluate_model(
                    model, dataset, device, cfg.batch_size, cfg.eval_iters,
                    effective_target_density,
                )
                eval_flops, eval_flops_relative = execution_flop_metrics(model, last_eval)
                eval_row = {
                    "split": "validation", "target_density": effective_target_density,
                    "estimated_block_flops": eval_flops,
                    "estimated_flops_vs_full_dense": eval_flops_relative,
                    **last_eval,
                }
                eval_row.update({f"layer_{i}_density": v for i, v in enumerate(last_eval["layer_utilization"])})
                eval_row.update({f"layer_{i}_soft_probability": v for i, v in enumerate(last_eval["layer_soft_probability"])})
                if model.config.model_type == "mor":
                    eval_row.update(
                        {
                            f"recursion_{i + 1}_utilization": value
                            for i, value in enumerate(last_eval["recursion_utilization"])
                        }
                    )
                    eval_row.update(
                        {
                            f"recursion_{i + 1}_soft_utilization": value
                            for i, value in enumerate(
                                last_eval["recursion_soft_utilization"]
                            )
                        }
                    )
                logger.log(eval_row, step)
                if model.config.model_type == "mor":
                    oracle_eval = evaluate_model(
                        model,
                        dataset,
                        device,
                        cfg.batch_size,
                        cfg.eval_iters,
                        effective_target_density,
                        routing_mode="topk",
                    )
                    oracle_flops, oracle_relative = execution_flop_metrics(model, oracle_eval)
                    oracle_row = {
                        "split": "validation_oracle_topk",
                        "target_density": effective_target_density,
                        "estimated_block_flops": oracle_flops,
                        "estimated_flops_vs_full_dense": oracle_relative,
                        **oracle_eval,
                    }
                    oracle_row.update(
                        {
                            f"recursion_{i + 1}_utilization": value
                            for i, value in enumerate(oracle_eval["recursion_utilization"])
                        }
                    )
                    oracle_row.update(
                        {
                            f"recursion_{i + 1}_soft_utilization": value
                            for i, value in enumerate(
                                oracle_eval["recursion_soft_utilization"]
                            )
                        }
                    )
                    logger.log(oracle_row, step)
                checkpoints = experiment_dir / "checkpoints"
                score = last_eval["val_loss"] + cfg.quality_compute_alpha * eval_flops_relative
                if last_eval["val_loss"] < best["val_loss"]:
                    best["val_loss"] = last_eval["val_loss"]
                    save_named_checkpoint(checkpoints / "best_val_loss.pt", model, optimizer, scheduler, step + 1, dataset, full_config, best)
                if last_eval["val_perplexity"] < best["val_perplexity"]:
                    best["val_perplexity"] = last_eval["val_perplexity"]
                    save_named_checkpoint(checkpoints / "best_val_perplexity.pt", model, optimizer, scheduler, step + 1, dataset, full_config, best)
                if score < best["quality_compute_score"]:
                    best["quality_compute_score"] = score
                    save_named_checkpoint(checkpoints / "best_quality_compute.pt", model, optimizer, scheduler, step + 1, dataset, full_config, best)
                save_named_checkpoint(checkpoints / "latest.pt", model, optimizer, scheduler, step + 1, dataset, full_config, best)
                layers = " ".join(f"L{i}:{v:.2f}" for i, v in enumerate(last_eval["layer_utilization"]))
                print(
                    f"validation | ce {last_eval['val_loss']:.4f} | ppl {last_eval['val_perplexity']:.3f} "
                    f"| acc {last_eval['val_accuracy']:.3f} | depth {last_eval['layers_per_token']:.2f}/"
                    f"{model.config.n_layers} | skip {100*last_eval['skip_fraction']:.1f}% | {layers}"
                )
    finally:
        logger.close()

    training_seconds = time.perf_counter() - training_started
    dense_flops = estimate_dense_block_flops(model.config, model.config.context_length)
    executed_flops, flops_relative = execution_flop_metrics(model, last_eval)
    summary = {
        "model": model.config.model_type,
        "router_type": (
            "expert_linear" if model.config.model_type == "mor"
            else model.config.router_type if model.config.model_type == "sparse"
            else "none"
        ),
        "training_method": "supervised",
        "seed": cfg.seed,
        "target_density": effective_target_density,
        "lambda_density": cfg.lambda_density if model.config.model_type == "sparse" else 0.0,
        "paper_reproduction": model.config.paper_reproduction,
        "mor_reproduction": model.config.mor_reproduction,
        "experiment_family": "mor_comparison" if model.config.model_type == "mor" else None,
        "n_layers": model.config.n_layers,
        "effective_target_depth": effective_target_density * model.config.n_layers,
        "learning_rate": cfg.learning_rate,
        "sparse_inference": model.config.sparse_inference,
        "density_reduction": cfg.density_reduction,
        "optimizer_name": cfg.optimizer_name,
        "parameter_count": model.parameter_count(),
        "active_parameter_estimate": (
            model.parameter_count()
            if model.config.model_type == "mor"
            else active_parameter_estimate(model, last_eval["compute_fraction"])
        ),
        "training_time_sec": training_seconds,
        "seconds_per_step": measured_step_seconds / max(cfg.max_steps - start_step, 1),
        "tokens_per_sec": (cfg.max_steps - start_step) * cfg.batch_size * model.config.context_length / max(training_seconds, 1e-9),
        "dense_block_flops_per_sequence": dense_flops,
        "estimated_executed_block_flops_per_sequence": executed_flops,
        "estimated_flops_vs_full_dense": flops_relative,
        "recursion_steps": model.config.recursion_steps if model.config.model_type == "mor" else None,
        "recursion_block_layers": (
            model.config.recursion_block_layers if model.config.model_type == "mor" else None
        ),
        "mor_capacity_factors": (
            model.config.mor_capacity_factors if model.config.model_type == "mor" else None
        ),
        "mor_aux_loss_coefficient": (
            model.config.mor_aux_loss_coefficient if model.config.model_type == "mor" else None
        ),
        "checkpoint": str(experiment_dir / "checkpoints" / "best_val_loss.pt"),
        **last_eval,
    }
    write_json(experiment_dir / "summary.json", summary)
    save_named_checkpoint(
        experiment_dir / "checkpoints" / "latest.pt", model, optimizer, scheduler,
        cfg.max_steps, dataset, full_config, best, summary,
    )
    print(f"completed experiment; artifacts saved in {experiment_dir}")


if __name__ == "__main__":
    main()
