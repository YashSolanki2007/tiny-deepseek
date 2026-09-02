"""Train the larger MLA+RoPE SkipLayer+MoE+MTP model on arithmetic and GSM8K."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from config import ModelConfig
from logging_utils import StructuredLogger
from losses import density_loss, scheduled_coefficient
from math_data import MathData
from math_training_utils import evaluate_math_answers, evaluate_math_model
from model import SparseMoEMTPTransformer, build_model
from utils import (
    gradient_norm,
    routing_metrics,
    save_checkpoint,
    select_device,
    set_seed,
    synchronize_device,
    top1_accuracy,
    write_json,
)


FIELDS = [
    "step", "split", "stage", "train_loss", "train_accuracy", "mtp_loss",
    "mtp_accuracy", "density_loss", "density_coefficient", "moe_aux_loss",
    "moe_router_entropy", "total_loss", "learning_rate", "gradient_norm",
    "seconds_per_step", "tokens_per_second", "layers_per_token",
    "compute_fraction", "skip_fraction", "routing_entropy",
    "expert_utilization_cv", "val_loss", "val_perplexity", "val_accuracy",
    "estimated_flops_vs_full_dense", "answer_exact_match", "answer_parse_rate",
    "generation_layers_per_token",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", default="experiments/math_grpo_seed42/supervised")
    parser.add_argument("--data-dir", default="data/gsm8k")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--synthetic-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--num-experts", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--target-density", type=float, default=0.70)
    parser.add_argument("--lambda-density", type=float, default=0.10)
    parser.add_argument("--mtp-weight", type=float, default=0.30)
    parser.add_argument("--moe-aux-weight", type=float, default=0.0001)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-iters", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--answer-eval-examples", type=int, default=4)
    parser.add_argument("--answer-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 <= args.synthetic_steps <= args.max_steps:
        raise ValueError("synthetic-steps must lie between zero and max-steps")
    device = select_device(args.device)
    set_seed(args.seed)
    data = MathData(args.data_dir, args.context_length, seed=args.seed)
    config = ModelConfig(
        vocab_size=data.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=0.05,
        model_type="sparse_moe_mtp_mla",
        router_type="linear",
        router_dim=32,
        initial_execute_probability=0.90,
        router_input_norm=True,
        router_bias=True,
        paper_reproduction=False,
        sparse_inference=True,
        tie_weights=True,
        moe_num_experts=args.num_experts,
        moe_top_k=args.top_k,
        moe_expert_d_ff=args.d_ff // args.top_k,
        moe_bias_update_speed=0.001,
        moe_aux_loss_coefficient=args.moe_aux_weight,
        mtp_loss_coefficient=args.mtp_weight,
        attention_type="mla",
        position_embedding_type="rope",
        mla_q_lora_rank=0,
        mla_kv_lora_rank=args.d_model // 4,
        mla_qk_nope_head_dim=args.d_model // args.n_heads // 2,
        mla_qk_rope_head_dim=args.d_model // args.n_heads // 2,
        mla_v_head_dim=args.d_model // args.n_heads,
    )
    model = build_model(config).to(device)
    if not isinstance(model, SparseMoEMTPTransformer):
        raise TypeError("math model must be SparseMoEMTPTransformer")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.max_steps, 1), eta_min=args.min_lr
    )
    experiment_dir = Path(args.experiment_dir)
    (experiment_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    logger = StructuredLogger(experiment_dir, FIELDS)
    run_config = {
        "stage": "math_supervised_curriculum",
        "model": config.to_dict(),
        "training": vars(args),
        "dataset": {
            "name": "GSM8K plus deterministic synthetic arithmetic",
            "train_examples": len(data.train_examples),
            "validation_examples": len(data.validation_examples),
            "test_examples": len(data.test_examples),
            "synthetic_train_examples": len(data.synthetic_train_examples),
            "tokenizer": "lossless UTF-8 bytes with BOS/EOS/PAD",
        },
        "device": str(device),
    }
    write_json(experiment_dir / "config.json", run_config)
    best_loss = float("inf")
    latest_eval: dict[str, float] = {}
    total_seconds = 0.0
    print(
        f"device={device} math-SFT parameters={model.parameter_count():,} "
        f"context={config.context_length} synthetic_steps={args.synthetic_steps}"
    )
    try:
        for step in range(args.max_steps):
            model.train()
            source = "synthetic" if step < args.synthetic_steps else "mixed"
            x, y = data.get_batch(source, args.batch_size, device)
            synchronize_device(device)
            started = time.perf_counter()
            output = model(x, y, routing_mode="gumbel")
            density_coefficient = scheduled_coefficient(
                step, args.max_steps, args.lambda_density, start=0.25, end=0.60
            )
            density = density_loss(output.hard_gates, args.target_density)
            total = (
                output.lm_loss
                + config.mtp_loss_coefficient * output.mtp_loss
                + config.moe_aux_loss_coefficient * output.moe_aux_loss
                + density_coefficient * density
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad = gradient_norm(model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_moe_selection_biases()
            scheduler.step()
            synchronize_device(device)
            seconds = time.perf_counter() - started
            total_seconds += seconds
            route = routing_metrics(output, config.n_layers)
            row = {
                "split": "train",
                "stage": source,
                "train_loss": float(output.lm_loss.item()),
                "train_accuracy": float(top1_accuracy(output.logits, y).item()),
                "mtp_loss": float(output.mtp_loss.item()),
                "mtp_accuracy": float(output.mtp_accuracy.item()),
                "density_loss": float(density.item()),
                "density_coefficient": density_coefficient,
                "moe_aux_loss": float(output.moe_aux_loss.item()),
                "moe_router_entropy": float(output.moe_router_entropy.item()),
                "total_loss": float(total.item()),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": grad,
                "seconds_per_step": seconds,
                "tokens_per_second": x.numel() / max(seconds, 1e-9),
                **{key: route[key] for key in (
                    "layers_per_token", "compute_fraction", "skip_fraction",
                    "routing_entropy", "expert_utilization_cv",
                )},
            }
            if step % args.log_interval == 0 or step == args.max_steps - 1:
                logger.log(row, step)
                print(
                    f"step {step:4d} {source:9s} | CE {row['train_loss']:.3f} "
                    f"| acc {row['train_accuracy']:.3f} | MTP {row['mtp_loss']:.3f} "
                    f"| depth {row['layers_per_token']:.2f}/{config.n_layers} | {seconds:.2f}s"
                )
            if (step + 1) % args.eval_interval == 0 or step == args.max_steps - 1:
                latest_eval = evaluate_math_model(
                    model, data, device, args.batch_size, args.eval_iters
                )
                logger.log({"split": "validation", "stage": source, **latest_eval}, step)
                if latest_eval["val_loss"] < best_loss:
                    best_loss = latest_eval["val_loss"]
                    save_checkpoint(
                        experiment_dir / "checkpoints" / "best_val_loss.pt",
                        model=model, optimizer=optimizer, scheduler=scheduler, step=step + 1,
                        stoi=data.stoi, itos=data.itos, training_config=run_config,
                        best_metrics={"val_loss": best_loss},
                    )
                save_checkpoint(
                    experiment_dir / "checkpoints" / "latest.pt",
                    model=model, optimizer=optimizer, scheduler=scheduler, step=step + 1,
                    stoi=data.stoi, itos=data.itos, training_config=run_config,
                    best_metrics={"val_loss": best_loss},
                )
                print(
                    f"validation | CE {latest_eval['val_loss']:.3f} "
                    f"| acc {latest_eval['val_accuracy']:.3f} "
                    f"| depth {latest_eval['layers_per_token']:.2f} "
                    f"| FLOPs {latest_eval['estimated_flops_vs_full_dense']:.3f}x"
                )
    finally:
        logger.close()

    answer_metrics, samples = evaluate_math_answers(
        model, data, device, split="validation", count=args.answer_eval_examples,
        max_new_tokens=args.answer_tokens, seed=args.seed,
    )
    summary = {
        "stage": "math_supervised_curriculum",
        "parameter_count": model.parameter_count(),
        "training_time_sec": total_seconds,
        **latest_eval,
        **answer_metrics,
        "samples": samples,
        "checkpoint": str(experiment_dir / "checkpoints" / "latest.pt"),
    }
    write_json(experiment_dir / "summary.json", summary)
    write_json(experiment_dir / "samples.json", samples)
    print(
        f"completed math SFT | exact {answer_metrics['answer_exact_match']:.3f} "
        f"| parse {answer_metrics['answer_parse_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
