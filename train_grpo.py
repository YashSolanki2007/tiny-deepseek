"""GRPO fine-tuning of a supervised GRU routing policy."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F

from config import GRPOConfig
from data import TinyShakespeareData
from evaluation import evaluate_model
from logging_utils import StructuredLogger
from losses import clipped_grpo_loss, grpo_reward, group_relative_advantages
from model import SparseDepthTransformer
from utils import (
    gradient_norm,
    load_checkpoint,
    restore_training_state,
    routing_metrics,
    save_checkpoint,
    select_device,
    set_seed,
    synchronize_device,
    write_json,
)


GRPO_FIELDS = [
    "step", "split", "mean_reward", "reward_std", "best_group_reward",
    "worst_group_reward", "mean_ce_component", "mean_compute_penalty", "mean_kl",
    "mean_advantage", "advantage_std", "policy_loss", "total_policy_loss",
    "routing_entropy", "clip_fraction", "mean_probability_ratio", "learning_rate",
    "gradient_norm", "router_gradient_norm", "transformer_gradient_norm",
    "tokens_processed", "seconds_per_step", "tokens_per_second", "val_loss",
    "val_perplexity", "val_accuracy", "mean_soft_gate", "mean_hard_gate",
    "layers_per_token", "compute_fraction", "skip_fraction", "validation_time_sec",
]


def policy_kl(current_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    """Mean decision KL for each trajectory; inputs are [N,T,L,2]."""
    current_log = F.log_softmax(current_logits, dim=-1)
    reference_log = F.log_softmax(reference_logits, dim=-1)
    current_probability = current_log.exp()
    return (current_probability * (current_log - reference_log)).sum(dim=-1).mean(dim=(1, 2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Supervised GRU best checkpoint")
    parser.add_argument("--resume", help="Resume a GRPO checkpoint")
    parser.add_argument("--experiment-dir")
    parser.add_argument("--data-path", default="data/input.txt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--policy-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--transformer-lr-scale", type=float, default=0.1)
    parser.add_argument("--lambda-compute-grpo", type=float, default=0.1)
    parser.add_argument("--beta-kl", type=float, default=0.01)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=10)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--grpo-router-only", action="store_true", default=True)
    modes.add_argument("--grpo-unfreeze-transformer", action="store_true")
    parser.add_argument("--kl-in-reward", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.checkpoint and not args.resume:
        raise SystemExit("Provide --checkpoint for a new run or --resume")
    device = select_device(args.device)
    set_seed(args.seed)
    source = args.resume or args.checkpoint
    model, checkpoint = load_checkpoint(source, device)
    if not isinstance(model, SparseDepthTransformer) or model.config.router_type != "gru":
        raise ValueError("GRPO requires a supervised sparse model with the GRU router")
    dataset = TinyShakespeareData(args.data_path, model.config.context_length)
    if dataset.stoi != checkpoint["stoi"]:
        raise ValueError("Checkpoint vocabulary differs from dataset")

    router_only = not args.grpo_unfreeze_transformer
    cfg = GRPOConfig(
        max_steps=args.max_steps, batch_size=args.batch_size, group_size=args.group_size,
        policy_epochs=args.policy_epochs, learning_rate=args.learning_rate,
        transformer_lr_scale=args.transformer_lr_scale,
        lambda_compute_grpo=args.lambda_compute_grpo, beta_kl=args.beta_kl,
        clip_epsilon=args.clip_epsilon, grad_clip=args.grad_clip,
        eval_interval=args.eval_interval, eval_iters=args.eval_iters,
        log_interval=args.log_interval, router_only=router_only,
        kl_in_reward=args.kl_in_reward, seed=args.seed,
    )
    reference_router = copy.deepcopy(model.router).to(device).eval()
    reference_state = checkpoint.get("reference_router_state")
    if reference_state:
        reference_router.load_state_dict(reference_state)
    for parameter in reference_router.parameters():
        parameter.requires_grad_(False)

    for parameter in model.parameters():
        parameter.requires_grad_(not router_only)
    for parameter in model.router.parameters():
        parameter.requires_grad_(True)
    router_params = list(model.router.parameters())
    router_ids = {id(p) for p in router_params}
    transformer_params = [p for p in model.parameters() if id(p) not in router_ids]
    groups = [{"params": router_params, "lr": cfg.learning_rate}]
    if not router_only:
        groups.append({"params": transformer_params, "lr": cfg.learning_rate * cfg.transformer_lr_scale})
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.max_steps, 1), eta_min=cfg.learning_rate * 0.1
    )
    start_step = 0
    best = {
        "val_loss": float("inf"), "val_perplexity": float("inf"),
        "quality_compute_score": float("inf"),
    }
    if args.resume:
        start_step = restore_training_state(checkpoint, optimizer, scheduler)
        best.update(checkpoint.get("best_metrics") or {})

    supervised_dir = Path(args.checkpoint).resolve().parent.parent if args.checkpoint else Path(source).resolve().parent.parent
    source_config = checkpoint.get("training_config") or {}
    target_density = float(
        source_config.get(
            "target_density", source_config.get("training", {}).get("target_density", 0.5)
        )
    )
    lambda_density = source_config.get(
        "lambda_density", source_config.get("training", {}).get("lambda_density")
    )
    default_dir = supervised_dir.parent / f"{supervised_dir.name}_grpo_lam{cfg.lambda_compute_grpo:g}_seed{cfg.seed}"
    experiment_dir = Path(args.experiment_dir) if args.experiment_dir else default_dir
    for child in ("checkpoints", "plots", "samples", "routing_visualizations"):
        (experiment_dir / child).mkdir(parents=True, exist_ok=True)
    before_metrics = source_config.get("before_grpo") if args.resume else None
    if not before_metrics:
        before_metrics = evaluate_model(
            model, dataset, device, cfg.batch_size, cfg.eval_iters, target_density
        )
    full_config = {
        "stage": "grpo", "model": model.config.to_dict(), "grpo": cfg.to_dict(),
        "device": str(device), "supervised_checkpoint": str(args.checkpoint or source),
        "target_density": target_density, "before_grpo": before_metrics,
        "old_policy_state": "Old action log-probabilities are retained per rollout; checkpoints occur only between rollouts.",
    }
    write_json(experiment_dir / "config.json", full_config)
    logger = StructuredLogger(
        experiment_dir, GRPO_FIELDS, purge_step=start_step if args.resume else None
    )
    latest_train: Dict[str, Any] = {}
    latest_eval = before_metrics
    tokens_processed = start_step * cfg.batch_size * cfg.group_size * model.config.context_length
    measured_step_seconds = 0.0
    training_started = time.perf_counter()
    print(
        f"device={device} GRPO router_only={router_only} group={cfg.group_size} "
        f"lambda_compute={cfg.lambda_compute_grpo} beta_kl={cfg.beta_kl}"
    )
    try:
        for step in range(start_step, cfg.max_steps):
            model.eval()  # keep dropout off; routing remains stochastic via routing_mode=sample
            synchronize_device(device)
            step_started = time.perf_counter()
            base_x, base_y = dataset.get_batch("train", cfg.batch_size, device)
            x = base_x.repeat_interleave(cfg.group_size, dim=0)
            y = base_y.repeat_interleave(cfg.group_size, dim=0)
            with torch.no_grad():
                sampled = model(x, y, routing_mode="sample")
                reference = model(
                    x, y, actions=sampled.actions, routing_mode="greedy",
                    router_override=reference_router,
                )
                sequence_ce = sampled.token_losses.mean(dim=1)
                compute = sampled.hard_gates.float().mean(dim=(1, 2))
                sampled_kl = policy_kl(sampled.route_logits, reference.route_logits)
                reward_kl_weight = cfg.beta_kl if cfg.kl_in_reward else 0.0
                rewards = grpo_reward(
                    sequence_ce, compute, sampled_kl,
                    cfg.lambda_compute_grpo, reward_kl_weight,
                )
                reward_groups = rewards.view(cfg.batch_size, cfg.group_size)
                advantages = group_relative_advantages(reward_groups).reshape(-1)
                old_log_probability = sampled.action_log_probs.mean(dim=(1, 2)).detach()
                actions = sampled.actions.detach()

            policy_losses, kls, entropies, ratios, clips = [], [], [], [], []
            for _ in range(cfg.policy_epochs):
                current = model(x, y, actions=actions, routing_mode="greedy")
                with torch.no_grad():
                    reference = model(
                        x, y, actions=actions, routing_mode="greedy",
                        router_override=reference_router,
                    )
                new_log_probability = current.action_log_probs.mean(dim=(1, 2))
                policy_loss, mean_ratio, clip_fraction = clipped_grpo_loss(
                    new_log_probability, old_log_probability, advantages, cfg.clip_epsilon
                )
                kl = policy_kl(current.route_logits, reference.route_logits).mean()
                total_policy_loss = policy_loss + cfg.beta_kl * kl
                optimizer.zero_grad(set_to_none=True)
                total_policy_loss.backward()
                router_grad = gradient_norm(router_params)
                transformer_grad = gradient_norm(transformer_params)
                total_grad = gradient_norm(p for p in model.parameters() if p.requires_grad)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.grad_clip
                )
                optimizer.step()
                policy_losses.append(policy_loss.detach())
                kls.append(kl.detach())
                entropies.append(current.routing_entropy.mean().detach())
                ratios.append(mean_ratio.detach())
                clips.append(clip_fraction.detach())
            scheduler.step()
            synchronize_device(device)
            step_seconds = time.perf_counter() - step_started
            measured_step_seconds += step_seconds
            tokens_processed += x.numel()
            latest_train = {
                "split": "grpo_train",
                "mean_reward": rewards.mean().item(),
                "reward_std": rewards.std(unbiased=False).item(),
                "best_group_reward": reward_groups.max(dim=1).values.mean().item(),
                "worst_group_reward": reward_groups.min(dim=1).values.mean().item(),
                "mean_ce_component": sequence_ce.mean().item(),
                "mean_compute_penalty": (cfg.lambda_compute_grpo * compute).mean().item(),
                "mean_kl": torch.stack(kls).mean().item(),
                "mean_advantage": advantages.mean().item(),
                "advantage_std": advantages.std(unbiased=False).item(),
                "policy_loss": torch.stack(policy_losses).mean().item(),
                "total_policy_loss": (torch.stack(policy_losses).mean() + cfg.beta_kl * torch.stack(kls).mean()).item(),
                "routing_entropy": torch.stack(entropies).mean().item(),
                "clip_fraction": torch.stack(clips).mean().item(),
                "mean_probability_ratio": torch.stack(ratios).mean().item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": total_grad, "router_gradient_norm": router_grad,
                "transformer_gradient_norm": transformer_grad,
                "tokens_processed": tokens_processed, "seconds_per_step": step_seconds,
                "tokens_per_second": x.numel() / max(step_seconds, 1e-9),
            }
            if step % cfg.log_interval == 0 or step == cfg.max_steps - 1:
                logger.log(latest_train, step)
                print(
                    f"step {step:5d} | reward {latest_train['mean_reward']:.4f}±"
                    f"{latest_train['reward_std']:.4f} | ce {latest_train['mean_ce_component']:.4f} "
                    f"| compute {compute.mean().item():.3f} | kl {latest_train['mean_kl']:.5f} "
                    f"| entropy {latest_train['routing_entropy']:.3f} | clip {latest_train['clip_fraction']:.3f}"
                )

            if (step + 1) % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
                latest_eval = evaluate_model(
                    model, dataset, device, cfg.batch_size, cfg.eval_iters, target_density
                )
                logger.log({"split": "validation", **latest_eval}, step)
                score = latest_eval["val_loss"] + cfg.lambda_compute_grpo * latest_eval["compute_fraction"]
                checkpoint_kwargs = dict(
                    model=model, optimizer=optimizer, scheduler=scheduler, step=step + 1,
                    stoi=dataset.stoi, itos=dataset.itos, training_config=full_config,
                    best_metrics=best, reference_router_state=reference_router.state_dict(),
                )
                if latest_eval["val_loss"] < best["val_loss"]:
                    best["val_loss"] = latest_eval["val_loss"]
                    save_checkpoint(experiment_dir / "checkpoints" / "best_val_loss.pt", **checkpoint_kwargs)
                if latest_eval["val_perplexity"] < best["val_perplexity"]:
                    best["val_perplexity"] = latest_eval["val_perplexity"]
                    save_checkpoint(
                        experiment_dir / "checkpoints" / "best_val_perplexity.pt",
                        **checkpoint_kwargs,
                    )
                if score < best["quality_compute_score"]:
                    best["quality_compute_score"] = score
                    save_checkpoint(experiment_dir / "checkpoints" / "best_quality_compute.pt", **checkpoint_kwargs)
                save_checkpoint(experiment_dir / "checkpoints" / "latest.pt", **checkpoint_kwargs)
                print(
                    f"validation | ce {latest_eval['val_loss']:.4f} | ppl {latest_eval['val_perplexity']:.3f} "
                    f"| acc {latest_eval['val_accuracy']:.3f} | depth {latest_eval['layers_per_token']:.2f}/"
                    f"{model.config.n_layers} | skip {100*latest_eval['skip_fraction']:.1f}%"
                )
    finally:
        logger.close()

    training_seconds = time.perf_counter() - training_started
    block_parameters = sum(
        parameter.numel() for block in model.blocks for parameter in block.parameters()
    )
    router_parameters = sum(parameter.numel() for parameter in model.router.parameters())
    always_active = model.parameter_count() - block_parameters - router_parameters
    active_parameters = (
        always_active + router_parameters + latest_eval["compute_fraction"] * block_parameters
    )
    summary = {
        "model": "sparse", "router_type": "gru", "training_method": "grpo",
        "seed": cfg.seed, "target_density": target_density,
        "lambda_density": lambda_density,
        "lambda_grpo": cfg.lambda_compute_grpo, "beta_kl": cfg.beta_kl,
        "parameter_count": model.parameter_count(),
        "active_parameter_estimate": active_parameters,
        "training_time_sec": training_seconds,
        "tokens_per_sec": (tokens_processed - start_step * cfg.batch_size * cfg.group_size * model.config.context_length) / max(training_seconds, 1e-9),
        "checkpoint": str(experiment_dir / "checkpoints" / "best_val_loss.pt"),
        "before_grpo": before_metrics, **latest_eval, **latest_train,
        "seconds_per_step": measured_step_seconds / max(cfg.max_steps - start_step, 1),
    }
    write_json(experiment_dir / "summary.json", summary)
    save_checkpoint(
        path=experiment_dir / "checkpoints" / "latest.pt", model=model,
        optimizer=optimizer, scheduler=scheduler, step=cfg.max_steps,
        stoi=dataset.stoi, itos=dataset.itos, training_config=full_config,
        best_metrics=best, summary=summary,
        reference_router_state=reference_router.state_dict(),
    )
    print(f"completed GRPO experiment; artifacts saved in {experiment_dir}")


if __name__ == "__main__":
    main()
