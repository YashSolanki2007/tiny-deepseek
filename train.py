"""Train a dense or dynamic-depth character Transformer."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict

import torch

from config import ModelConfig, TrainConfig
from data import TinyShakespeareData
from model import TransformerBase, build_model
from utils import (
    compute_penalty,
    estimate_dense_block_flops,
    get_compute_lambda,
    perplexity,
    routing_metrics,
    save_checkpoint,
    select_device,
    set_seed,
    write_json,
)


def learning_rate(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(cfg.warmup_steps, 1)
    decay_steps = max(cfg.max_steps - cfg.warmup_steps, 1)
    ratio = min(max((step - cfg.warmup_steps) / decay_steps, 0.0), 1.0)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coefficient * (cfg.learning_rate - cfg.min_lr)


@torch.inference_mode()
def evaluate(
    model: TransformerBase,
    dataset: TinyShakespeareData,
    cfg: TrainConfig,
    device: torch.device,
    current_lambda: float,
) -> Dict[str, Any]:
    model.eval()
    lm_losses = []
    total_losses = []
    penalties = []
    hard_sum = torch.zeros(model.config.n_layers)
    soft_sum = torch.zeros(model.config.n_layers)

    for _ in range(cfg.eval_iters):
        x, y = dataset.get_batch("val", cfg.batch_size, device)
        output = model(x, y)
        penalty = (
            compute_penalty(output.soft_gates, cfg.compute_loss, cfg.target_compute)
            if output.soft_gates is not None
            else output.lm_loss.new_zeros(())
        )
        total = output.lm_loss + current_lambda * penalty
        metrics = routing_metrics(output, model.config.n_layers)
        lm_losses.append(output.lm_loss.item())
        total_losses.append(total.item())
        penalties.append(penalty.item())
        hard_sum += torch.tensor(metrics["layer_utilization"])
        soft_sum += torch.tensor(metrics["layer_soft_probability"])

    layer_hard = hard_sum / cfg.eval_iters
    layer_soft = soft_sum / cfg.eval_iters
    compute_fraction = layer_hard.mean().item()
    val_loss = sum(lm_losses) / len(lm_losses)
    model.train()
    return {
        "val_loss": val_loss,
        "val_perplexity": perplexity(val_loss),
        "val_total_loss": sum(total_losses) / len(total_losses),
        "compute_loss": sum(penalties) / len(penalties),
        "mean_soft_gate": layer_soft.mean().item(),
        "mean_hard_gate": compute_fraction,
        "layers_per_token": compute_fraction * model.config.n_layers,
        "compute_fraction": compute_fraction,
        "skip_fraction": 1.0 - compute_fraction,
        "layer_utilization": layer_hard.tolist(),
        "layer_soft_probability": layer_soft.tolist(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["dense", "dynamic"], default="dynamic")
    parser.add_argument("--router", choices=["gru", "mlp"], default="gru")
    parser.add_argument("--data-path", default="data/input.txt")
    parser.add_argument("--output-dir", default="runs/dynamic_lam0.01")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--router-dim", type=int, default=32)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--gate-bias", type=float, default=2.2)
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
    parser.add_argument("--lambda-compute", type=float, default=0.01)
    parser.add_argument("--compute-loss", choices=["linear", "target"], default="linear")
    parser.add_argument("--target-compute", type=float, default=0.5)
    parser.add_argument("--compute-warmup-start", type=float, default=0.10)
    parser.add_argument("--compute-warmup-end", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = select_device(args.device)
    set_seed(args.seed)
    dataset = TinyShakespeareData(args.data_path, args.context_length)

    model_cfg = ModelConfig(
        vocab_size=dataset.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        model_type=args.model,
        router_type=args.router,
        router_dim=args.router_dim,
        gate_threshold=args.gate_threshold,
        gate_bias=args.gate_bias,
        tie_weights=not args.no_tie_weights,
    )
    train_cfg = TrainConfig(
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
        lambda_compute=args.lambda_compute,
        compute_loss=args.compute_loss,
        target_compute=args.target_compute,
        compute_warmup_start=args.compute_warmup_start,
        compute_warmup_end=args.compute_warmup_end,
        seed=args.seed,
    )
    model = build_model(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        weight_decay=train_cfg.weight_decay,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device={device} model={args.model} router={args.router} "
        f"parameters={model.parameter_count():,}"
    )

    started = time.perf_counter()
    tokens_seen = 0
    latest_eval: Dict[str, Any] = {}
    for step in range(train_cfg.max_steps):
        model.train()
        lr = learning_rate(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        current_lambda = get_compute_lambda(
            step,
            train_cfg.max_steps,
            train_cfg.lambda_compute,
            train_cfg.compute_warmup_start,
            train_cfg.compute_warmup_end,
        )
        x, y = dataset.get_batch("train", train_cfg.batch_size, device)
        output = model(x, y)
        penalty = (
            compute_penalty(output.soft_gates, train_cfg.compute_loss, train_cfg.target_compute)
            if output.soft_gates is not None
            else output.lm_loss.new_zeros(())
        )
        total_loss = output.lm_loss + current_lambda * penalty
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        tokens_seen += x.numel()

        if step % train_cfg.log_interval == 0 or step == train_cfg.max_steps - 1:
            route = routing_metrics(output, model.config.n_layers)
            print(
                f"step {step:5d} | loss {total_loss.item():.4f} | lm {output.lm_loss.item():.4f} "
                f"| compute_loss {penalty.item():.4f} | lambda {current_lambda:.5f} "
                f"| soft {route['mean_soft_gate']:.3f} | hard {route['mean_hard_gate']:.3f} "
                f"| depth {route['layers_per_token']:.2f}/{model.config.n_layers} "
                f"| lr {lr:.2e} | grad {float(grad_norm):.2f}"
            )

        if (step + 1) % train_cfg.eval_interval == 0 or step == train_cfg.max_steps - 1:
            latest_eval = evaluate(model, dataset, train_cfg, device, current_lambda)
            layer_text = " ".join(
                f"L{i}:{value:.2f}" for i, value in enumerate(latest_eval["layer_utilization"])
            )
            print(
                f"validation | ce {latest_eval['val_loss']:.4f} | "
                f"ppl {latest_eval['val_perplexity']:.3f} | "
                f"depth {latest_eval['layers_per_token']:.2f}/{model.config.n_layers} | "
                f"skip {100 * latest_eval['skip_fraction']:.1f}% | {layer_text}"
            )

    elapsed = time.perf_counter() - started
    summary: Dict[str, Any] = {
        "model": args.model,
        "router": args.router if args.model == "dynamic" else "none",
        "lambda": args.lambda_compute if args.model == "dynamic" else 0.0,
        "compute_loss_mode": args.compute_loss if args.model == "dynamic" else "none",
        "parameters": model.parameter_count(),
        "training_seconds": elapsed,
        "training_tokens_per_second": tokens_seen / max(elapsed, 1e-9),
        **latest_eval,
    }
    summary["ppl"] = summary["val_perplexity"]
    dense_flops = estimate_dense_block_flops(model.config, model.config.context_length)
    summary["dense_block_flops_per_sequence"] = dense_flops
    summary["estimated_executed_block_flops_per_sequence"] = (
        dense_flops * summary["compute_fraction"]
    )
    summary_path = output_dir / "summary.json"
    checkpoint_path = output_dir / "checkpoint.pt"
    write_json(summary_path, summary)
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        train_cfg.max_steps,
        dataset.stoi,
        dataset.itos,
        train_cfg.to_dict(),
        summary,
    )
    print(f"saved {checkpoint_path} and {summary_path}")


if __name__ == "__main__":
    main()
