"""Budget-guided GRPO fine-tuning for a paper-faithful SkipLayer router."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from data import TinyShakespeareData
from evaluation import evaluate_model
from logging_utils import StructuredLogger
from losses import clipped_grpo_loss_per_decision, grpo_reward, group_relative_advantages
from model import SparseDepthTransformer
from utils import (
    estimate_dense_block_flops,
    estimate_skiplayer_flops,
    gradient_norm,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
    select_device,
    set_seed,
    synchronize_device,
    write_json,
)


BASE_FIELDS = [
    "step", "split", "mean_reward", "reward_std", "best_group_reward",
    "worst_group_reward", "mean_ce_component", "mean_compute_penalty", "mean_kl",
    "mean_advantage", "advantage_std", "policy_loss", "total_policy_loss",
    "routing_entropy", "clip_fraction", "mean_probability_ratio", "learning_rate",
    "gradient_norm", "router_gradient_norm", "transformer_gradient_norm",
    "tokens_processed", "seconds_per_step", "tokens_per_second", "val_loss",
    "val_perplexity", "val_accuracy", "mean_soft_gate", "mean_hard_gate",
    "layers_per_token", "compute_fraction", "skip_fraction", "validation_time_sec",
    "estimated_block_flops", "estimated_flops_vs_dense",
]


def policy_kl(current_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    """Mean policy KL per trajectory for logits shaped [N,T,L,2]."""
    current_log = F.log_softmax(current_logits, dim=-1)
    reference_log = F.log_softmax(reference_logits, dim=-1)
    current_probability = current_log.exp()
    return (current_probability * (current_log - reference_log)).sum(dim=-1).mean(dim=(1, 2))


def add_flop_metrics(
    metrics: dict[str, Any], model: SparseDepthTransformer, target_density: float
) -> dict[str, Any]:
    density = float(metrics["compute_fraction"])
    sequence_flops = estimate_skiplayer_flops(
        model.config, model.config.context_length, density
    )
    effective_depth_dense = copy.copy(model.config)
    effective_depth_dense.n_layers = max(
        1, round(model.config.n_layers * target_density)
    )
    dense_flops = estimate_dense_block_flops(
        effective_depth_dense, model.config.context_length
    )
    return {
        **metrics,
        "estimated_executed_block_flops_per_sequence": sequence_flops,
        "estimated_block_flops": sequence_flops,
        "estimated_flops_vs_dense": sequence_flops / dense_flops,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Paper SkipLayer supervised checkpoint")
    parser.add_argument("--resume", help="Resume a budget-guided GRPO checkpoint")
    parser.add_argument("--experiment-dir")
    parser.add_argument("--data-path", default="data/input.txt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--depth-budgets", type=int, nargs="+", default=[3, 4, 5, 8])
    parser.add_argument("--exploration-epsilon", type=float, default=0.8)
    parser.add_argument("--policy-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-compute-grpo", type=float, default=0.1)
    parser.add_argument("--beta-kl", type=float, default=0.01)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.checkpoint and not args.resume:
        raise SystemExit("Provide --checkpoint for a new run or --resume")
    if not 0 <= args.exploration_epsilon < 1:
        raise ValueError("exploration-epsilon must be in [0,1); the anchor is set to 1 internally")

    device = select_device(args.device)
    set_seed(args.seed)
    source = args.resume or args.checkpoint
    model, checkpoint = load_checkpoint(source, device)
    if not isinstance(model, SparseDepthTransformer):
        raise ValueError("paper GRPO requires a sparse SkipLayer checkpoint")
    if model.config.router_type != "linear" or not model.config.paper_reproduction:
        raise ValueError("paper GRPO requires the paper-faithful bias-free linear router")
    budgets = list(args.depth_budgets)
    if len(budgets) < 2 or budgets[-1] != model.config.n_layers:
        raise ValueError("depth-budgets must end with the full physical depth quality anchor")
    if any(value < 0 or value > model.config.n_layers for value in budgets):
        raise ValueError("every depth budget must be between 0 and n_layers")
    if len(set(budgets)) != len(budgets):
        raise ValueError("depth budgets must be unique")

    dataset = TinyShakespeareData(args.data_path, model.config.context_length)
    if dataset.stoi != checkpoint["stoi"]:
        raise ValueError("checkpoint vocabulary differs from dataset")

    reference_router = copy.deepcopy(model.router).to(device).eval()
    reference_state = checkpoint.get("reference_router_state")
    if reference_state:
        reference_router.load_state_dict(reference_state)
    for parameter in reference_router.parameters():
        parameter.requires_grad_(False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.router.parameters():
        parameter.requires_grad_(True)
    router_params = list(model.router.parameters())
    router_ids = {id(parameter) for parameter in router_params}
    transformer_params = [
        parameter for parameter in model.parameters() if id(parameter) not in router_ids
    ]
    optimizer = torch.optim.AdamW(
        router_params, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.max_steps, 1), eta_min=args.learning_rate * 0.1
    )
    start_step = 0
    best = {
        "val_loss": float("inf"),
        "val_perplexity": float("inf"),
        "quality_compute_score": float("inf"),
    }
    if args.resume:
        start_step = restore_training_state(checkpoint, optimizer, scheduler)
        best.update(checkpoint.get("best_metrics") or {})

    supervised_dir = Path(args.checkpoint or source).resolve().parent.parent
    default_dir = supervised_dir.parent / (
        f"{supervised_dir.name}_budget_grpo_lam{args.lambda_compute_grpo:g}_seed{args.seed}"
    )
    experiment_dir = Path(args.experiment_dir) if args.experiment_dir else default_dir
    for child in ("checkpoints", "plots", "samples", "routing_visualizations"):
        (experiment_dir / child).mkdir(parents=True, exist_ok=True)

    source_config = checkpoint.get("training_config") or {}
    target_density = float(
        source_config.get(
            "target_density",
            source_config.get("training", {}).get(
                "target_density", 0.5
            ),
        )
    )
    lambda_density = source_config.get(
        "lambda_density", source_config.get("training", {}).get("lambda_density")
    )
    before_metrics = source_config.get("before_grpo") if args.resume else None
    if not before_metrics:
        before_metrics = evaluate_model(
            model, dataset, device, args.batch_size, args.eval_iters, target_density
        )
    before_metrics = add_flop_metrics(before_metrics, model, target_density)

    grpo_config = {
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "group_size": len(budgets),
        "depth_budgets": budgets,
        "exploration_epsilon": args.exploration_epsilon,
        "anchor_depth": budgets[-1],
        "anchor_in_policy_loss": False,
        "policy_epochs": args.policy_epochs,
        "learning_rate": args.learning_rate,
        "lambda_compute_grpo": args.lambda_compute_grpo,
        "beta_kl": args.beta_kl,
        "clip_epsilon": args.clip_epsilon,
        "grad_clip": args.grad_clip,
        "eval_interval": args.eval_interval,
        "eval_iters": args.eval_iters,
        "log_interval": args.log_interval,
        "router_only": True,
        "seed": args.seed,
    }
    full_config = {
        "stage": "grpo",
        "grpo_variant": "budget_guided_per_decision",
        "model": model.config.to_dict(),
        "grpo": grpo_config,
        "device": str(device),
        "supervised_checkpoint": str(args.checkpoint or source),
        "target_density": target_density,
        "before_grpo": before_metrics,
        "sampling_note": (
            "Non-anchor actions come from an epsilon mixture of the router and a remaining-budget "
            "controller; exact behavior probabilities are retained for per-decision PPO ratios. "
            "The deterministic full-depth anchor affects group-relative advantages but is excluded "
            "from the policy loss."
        ),
    }
    write_json(experiment_dir / "config.json", full_config)

    budget_fields = [
        f"budget_{budget}_{metric}"
        for budget in budgets
        for metric in ("depth", "ce", "reward", "flops_vs_dense")
    ]
    logger = StructuredLogger(
        experiment_dir,
        BASE_FIELDS + budget_fields,
        purge_step=start_step if args.resume else None,
    )
    latest_eval = before_metrics
    latest_train: dict[str, Any] = {}
    group_size = len(budgets)
    tokens_processed = start_step * args.batch_size * group_size * model.config.context_length
    measured_step_seconds = 0.0
    training_started = time.perf_counter()
    print(
        f"device={device} paper-GRPO budgets={budgets} epsilon={args.exploration_epsilon} "
        f"lambda_compute={args.lambda_compute_grpo} beta_kl={args.beta_kl}"
    )

    try:
        for step in range(start_step, args.max_steps):
            model.eval()
            synchronize_device(device)
            step_started = time.perf_counter()
            base_x, base_y = dataset.get_batch("train", args.batch_size, device)
            x = base_x.repeat_interleave(group_size, dim=0)
            y = base_y.repeat_interleave(group_size, dim=0)
            trajectory_budgets = torch.tensor(
                budgets, device=device, dtype=torch.float32
            ).repeat(args.batch_size)
            epsilons = torch.full(
                (x.shape[0],), args.exploration_epsilon, device=device
            )
            anchor_mask = trajectory_budgets.eq(float(model.config.n_layers))
            epsilons[anchor_mask] = 1.0
            trainable_mask = ~anchor_mask

            with torch.no_grad():
                sampled = model(
                    x,
                    y,
                    routing_mode="budget",
                    target_depths=trajectory_budgets,
                    exploration_epsilon=epsilons,
                )
                reference = model(
                    x,
                    y,
                    actions=sampled.actions,
                    routing_mode="greedy",
                    router_override=reference_router,
                )
                sequence_ce = sampled.token_losses.mean(dim=1)
                compute = sampled.hard_gates.float().mean(dim=(1, 2))
                sampled_kl = policy_kl(sampled.route_logits, reference.route_logits)
                rewards = grpo_reward(
                    sequence_ce,
                    compute,
                    sampled_kl,
                    args.lambda_compute_grpo,
                    0.0,
                )
                reward_groups = rewards.view(args.batch_size, group_size)
                advantages = group_relative_advantages(reward_groups).reshape(-1)
                behavior_log_probability = sampled.behavior_log_probs.detach()
                actions = sampled.actions.detach()

            policy_losses, kls, entropies, ratios, clips = [], [], [], [], []
            for _ in range(args.policy_epochs):
                model.train()
                current = model(x, y, actions=actions, routing_mode="greedy")
                with torch.no_grad():
                    reference = model(
                        x,
                        y,
                        actions=actions,
                        routing_mode="greedy",
                        router_override=reference_router,
                    )
                policy_loss, mean_ratio, clip_fraction = clipped_grpo_loss_per_decision(
                    current.action_log_probs,
                    behavior_log_probability,
                    advantages,
                    args.clip_epsilon,
                    trajectory_mask=trainable_mask,
                )
                kl = policy_kl(current.route_logits, reference.route_logits).mean()
                total_policy_loss = policy_loss + args.beta_kl * kl
                optimizer.zero_grad(set_to_none=True)
                total_policy_loss.backward()
                router_grad = gradient_norm(router_params)
                transformer_grad = gradient_norm(transformer_params)
                total_grad = gradient_norm(router_params)
                torch.nn.utils.clip_grad_norm_(router_params, args.grad_clip)
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
            dense_reference_config = copy.copy(model.config)
            dense_reference_config.n_layers = max(
                1, round(model.config.n_layers * target_density)
            )
            dense_flops = estimate_dense_block_flops(
                dense_reference_config, model.config.context_length
            )
            latest_train = {
                "split": "grpo_train",
                "mean_reward": rewards.mean().item(),
                "reward_std": rewards.std(unbiased=False).item(),
                "best_group_reward": reward_groups.max(dim=1).values.mean().item(),
                "worst_group_reward": reward_groups.min(dim=1).values.mean().item(),
                "mean_ce_component": sequence_ce.mean().item(),
                "mean_compute_penalty": (args.lambda_compute_grpo * compute).mean().item(),
                "mean_kl": torch.stack(kls).mean().item(),
                "mean_advantage": advantages.mean().item(),
                "advantage_std": advantages.std(unbiased=False).item(),
                "policy_loss": torch.stack(policy_losses).mean().item(),
                "total_policy_loss": (
                    torch.stack(policy_losses).mean() + args.beta_kl * torch.stack(kls).mean()
                ).item(),
                "routing_entropy": torch.stack(entropies).mean().item(),
                "clip_fraction": torch.stack(clips).mean().item(),
                "mean_probability_ratio": torch.stack(ratios).mean().item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": total_grad,
                "router_gradient_norm": router_grad,
                "transformer_gradient_norm": transformer_grad,
                "tokens_processed": tokens_processed,
                "seconds_per_step": step_seconds,
                "tokens_per_second": x.numel() / max(step_seconds, 1e-9),
            }
            for group_index, budget in enumerate(budgets):
                selector = torch.arange(group_index, x.shape[0], group_size, device=device)
                budget_compute = compute[selector].mean()
                budget_depth = budget_compute * model.config.n_layers
                budget_flops = estimate_skiplayer_flops(
                    model.config, model.config.context_length, budget_compute.item()
                )
                latest_train.update(
                    {
                        f"budget_{budget}_depth": budget_depth.item(),
                        f"budget_{budget}_ce": sequence_ce[selector].mean().item(),
                        f"budget_{budget}_reward": rewards[selector].mean().item(),
                        f"budget_{budget}_flops_vs_dense": budget_flops / dense_flops,
                    }
                )

            if step % args.log_interval == 0 or step == args.max_steps - 1:
                logger.log(latest_train, step)
                depths = ", ".join(
                    f"{budget}:{latest_train[f'budget_{budget}_depth']:.2f}" for budget in budgets
                )
                print(
                    f"step {step:5d} | reward {latest_train['mean_reward']:.4f}±"
                    f"{latest_train['reward_std']:.4f} | ce {latest_train['mean_ce_component']:.4f} "
                    f"| depths [{depths}] | kl {latest_train['mean_kl']:.5f} "
                    f"| clip {latest_train['clip_fraction']:.3f}"
                )

            if (step + 1) % args.eval_interval == 0 or step == args.max_steps - 1:
                latest_eval = evaluate_model(
                    model, dataset, device, args.batch_size, args.eval_iters, target_density
                )
                latest_eval = add_flop_metrics(latest_eval, model, target_density)
                logger.log({"split": "validation", **latest_eval}, step)
                score = latest_eval["val_loss"] + args.lambda_compute_grpo * latest_eval["compute_fraction"]
                checkpoint_kwargs = dict(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step + 1,
                    stoi=dataset.stoi,
                    itos=dataset.itos,
                    training_config=full_config,
                    best_metrics=best,
                    reference_router_state=reference_router.state_dict(),
                )
                if latest_eval["val_loss"] < best["val_loss"]:
                    best["val_loss"] = latest_eval["val_loss"]
                    save_checkpoint(
                        experiment_dir / "checkpoints" / "best_val_loss.pt", **checkpoint_kwargs
                    )
                if latest_eval["val_perplexity"] < best["val_perplexity"]:
                    best["val_perplexity"] = latest_eval["val_perplexity"]
                    save_checkpoint(
                        experiment_dir / "checkpoints" / "best_val_perplexity.pt",
                        **checkpoint_kwargs,
                    )
                if score < best["quality_compute_score"]:
                    best["quality_compute_score"] = score
                    save_checkpoint(
                        experiment_dir / "checkpoints" / "best_quality_compute.pt",
                        **checkpoint_kwargs,
                    )
                save_checkpoint(
                    experiment_dir / "checkpoints" / "latest.pt", **checkpoint_kwargs
                )
                print(
                    f"validation | ce {latest_eval['val_loss']:.4f} | "
                    f"ppl {latest_eval['val_perplexity']:.3f} | depth "
                    f"{latest_eval['layers_per_token']:.2f}/{model.config.n_layers} | "
                    f"skip {100 * latest_eval['skip_fraction']:.1f}% | "
                    f"FLOPs {latest_eval['estimated_flops_vs_dense']:.3f}x matched dense"
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
        always_active
        + router_parameters
        + latest_eval["compute_fraction"] * block_parameters
    )
    selected_checkpoint = experiment_dir / "checkpoints" / "best_quality_compute.pt"
    summary = {
        "model": "sparse",
        "router_type": "linear",
        "training_method": "grpo",
        "grpo_variant": "budget_guided_per_decision",
        "paper_reproduction": True,
        "seed": args.seed,
        "n_layers": model.config.n_layers,
        "target_density": target_density,
        "lambda_density": lambda_density,
        "lambda_grpo": args.lambda_compute_grpo,
        "beta_kl": args.beta_kl,
        "depth_budgets": budgets,
        "exploration_epsilon": args.exploration_epsilon,
        "parameter_count": model.parameter_count(),
        "active_parameter_estimate": active_parameters,
        "training_time_sec": training_seconds,
        "tokens_per_sec": (
            tokens_processed
            - start_step * args.batch_size * group_size * model.config.context_length
        ) / max(training_seconds, 1e-9),
        "checkpoint": str(selected_checkpoint),
        "before_grpo": before_metrics,
        **latest_eval,
        **latest_train,
        "seconds_per_step": measured_step_seconds / max(args.max_steps - start_step, 1),
    }
    write_json(experiment_dir / "summary.json", summary)
    save_checkpoint(
        path=experiment_dir / "checkpoints" / "latest.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=args.max_steps,
        stoi=dataset.stoi,
        itos=dataset.itos,
        training_config=full_config,
        best_metrics=best,
        summary=summary,
        reference_router_state=reference_router.state_dict(),
    )
    print(f"completed paper GRPO experiment; artifacts saved in {experiment_dir}")


if __name__ == "__main__":
    main()
