"""Budget-guided per-decision GRPO for a supervised Mixture-of-Recursions router."""

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
from model import MixtureOfRecursionsTransformer
from utils import (
    estimate_dense_block_flops,
    estimate_mor_flops,
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
    "estimated_block_flops", "estimated_flops_vs_full_dense", "mor_aux_loss",
    "mor_router_accuracy", "mean_recursions_per_token",
]


def masked_policy_kl(
    current_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    decision_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_log = F.log_softmax(current_logits, dim=-1)
    reference_log = F.log_softmax(reference_logits, dim=-1)
    current_probability = current_log.exp()
    decision_kl = (current_probability * (current_log - reference_log)).sum(dim=-1)
    valid = decision_mask.float()
    per_trajectory = (decision_kl * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1)
    return per_trajectory, decision_kl[decision_mask].mean()


def trajectory_flop_fraction(
    model: MixtureOfRecursionsTransformer, actions: torch.Tensor
) -> torch.Tensor:
    """Paper-style recursion-wise FLOPs for each trajectory."""
    config = model.config
    utilization = actions.float().mean(dim=1)
    length = float(config.context_length)
    d, ff = config.d_model, config.d_ff
    full_layer = 8 * length * d * d + 4 * length * length * d + 4 * length * d * ff
    total = utilization.new_full((utilization.shape[0],), 2 * full_layer)
    previous = torch.ones_like(total)
    for recursion_index in range(config.recursion_steps):
        active = torch.minimum(utilization[:, recursion_index], previous)
        projections = 8 * active * length * d * d
        attention = 4 * (active * length).square() * d
        feed_forward = 4 * active * length * d * ff
        total = total + config.recursion_block_layers * (
            projections + attention + feed_forward
        )
        total = total + 2 * previous * length * d
        previous = active
    dense = estimate_dense_block_flops(config, config.context_length)
    return total / dense


def add_flops(
    metrics: dict[str, Any], model: MixtureOfRecursionsTransformer
) -> dict[str, Any]:
    executed = estimate_mor_flops(
        model.config,
        model.config.context_length,
        metrics["recursion_utilization"],
    )
    dense = estimate_dense_block_flops(model.config, model.config.context_length)
    return {
        **metrics,
        "estimated_executed_block_flops_per_sequence": executed,
        "estimated_block_flops": executed,
        "estimated_flops_vs_full_dense": executed / dense,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--experiment-dir")
    parser.add_argument("--data-path", default="data/input.txt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--recursion-budgets", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--exploration-epsilon", type=float, default=0.8)
    parser.add_argument("--policy-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-compute-grpo", type=float, default=1.0)
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
        raise SystemExit("Provide --checkpoint or --resume")
    if not 0 <= args.exploration_epsilon < 1:
        raise ValueError("exploration-epsilon must be in [0,1)")
    device = select_device(args.device)
    set_seed(args.seed)
    source = args.resume or args.checkpoint
    model, checkpoint = load_checkpoint(source, device)
    if not isinstance(model, MixtureOfRecursionsTransformer):
        raise ValueError("MoR GRPO requires a supervised MoR checkpoint")
    budgets = list(args.recursion_budgets)
    if budgets[-1] != model.config.recursion_steps or len(set(budgets)) != len(budgets):
        raise ValueError("recursion budgets must be unique and end at the full recursion count")
    if any(value < 1 or value > model.config.recursion_steps for value in budgets):
        raise ValueError("recursion budgets must lie in [1, recursion_steps]")

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

    before = evaluate_model(
        model, dataset, device, args.batch_size, args.eval_iters, routing_mode="greedy"
    )
    before = add_flops(before, model)
    grpo_config = vars(args).copy()
    grpo_config.update(
        {
            "group_size": len(budgets),
            "anchor_recursions": budgets[-1],
            "anchor_in_policy_loss": False,
            "router_only": True,
        }
    )
    full_config = {
        "stage": "grpo",
        "grpo_variant": "mor_budget_guided_per_decision",
        "model": model.config.to_dict(),
        "grpo": grpo_config,
        "device": str(device),
        "supervised_checkpoint": str(args.checkpoint or source),
        "before_grpo": before,
        "sampling_note": (
            "Budgets select 1/2/3 hierarchical recursions through a recorded epsilon mixture. "
            "The full-recursion quality anchor is excluded from the policy loss."
        ),
    }
    write_json(experiment_dir / "config.json", full_config)

    budget_fields = [
        f"budget_{budget}_{metric}"
        for budget in budgets
        for metric in ("recursions", "depth", "ce", "reward", "flops_vs_full_dense")
    ]
    recursion_fields = [
        f"recursion_{index + 1}_{kind}"
        for index in range(model.config.recursion_steps)
        for kind in ("utilization", "soft_utilization")
    ]
    logger = StructuredLogger(
        experiment_dir,
        BASE_FIELDS + budget_fields + recursion_fields,
        purge_step=start_step if args.resume else None,
    )
    group_size = len(budgets)
    tokens_processed = start_step * args.batch_size * group_size * model.config.context_length
    latest_eval, latest_train = before, {}
    measured_step_seconds = 0.0
    training_started = time.perf_counter()
    print(
        f"device={device} MoR-GRPO budgets={budgets} epsilon={args.exploration_epsilon} "
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
            trajectory_budgets = torch.tensor(budgets, device=device).repeat(args.batch_size)
            epsilons = torch.full(
                (x.shape[0],), args.exploration_epsilon, device=device
            )
            anchor_mask = trajectory_budgets.eq(model.config.recursion_steps)
            epsilons[anchor_mask] = 1.0
            trainable_trajectories = ~anchor_mask

            with torch.no_grad():
                sampled = model(
                    x,
                    y,
                    routing_mode="budget",
                    target_recursions=trajectory_budgets,
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
                compute = trajectory_flop_fraction(model, sampled.actions)
                sampled_kl, _ = masked_policy_kl(
                    sampled.route_logits,
                    reference.route_logits,
                    sampled.routing_decision_mask,
                )
                rewards = grpo_reward(
                    sequence_ce, compute, sampled_kl, args.lambda_compute_grpo, 0.0
                )
                reward_groups = rewards.view(args.batch_size, group_size)
                advantages = group_relative_advantages(reward_groups).reshape(-1)
                behavior_log_probability = sampled.behavior_log_probs.detach()
                actions = sampled.actions.detach()
                decision_mask = sampled.routing_decision_mask.detach()

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
                    trajectory_mask=trainable_trajectories,
                    decision_mask=decision_mask,
                )
                _, kl = masked_policy_kl(
                    current.route_logits, reference.route_logits, decision_mask
                )
                total_policy_loss = policy_loss + args.beta_kl * kl
                optimizer.zero_grad(set_to_none=True)
                total_policy_loss.backward()
                router_grad = gradient_norm(router_params)
                transformer_grad = gradient_norm(transformer_params)
                torch.nn.utils.clip_grad_norm_(router_params, args.grad_clip)
                optimizer.step()
                policy_losses.append(policy_loss.detach())
                kls.append(kl.detach())
                entropies.append(current.routing_entropy[decision_mask].mean().detach())
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
                "gradient_norm": router_grad,
                "router_gradient_norm": router_grad,
                "transformer_gradient_norm": transformer_grad,
                "tokens_processed": tokens_processed,
                "seconds_per_step": step_seconds,
                "tokens_per_second": x.numel() / max(step_seconds, 1e-9),
            }
            for recursion_index, value in enumerate(sampled.recursion_utilization.tolist()):
                latest_train[f"recursion_{recursion_index + 1}_utilization"] = value
            for recursion_index, value in enumerate(sampled.recursion_soft_utilization.tolist()):
                latest_train[f"recursion_{recursion_index + 1}_soft_utilization"] = value
            for group_index, budget in enumerate(budgets):
                selector = torch.arange(group_index, x.shape[0], group_size, device=device)
                recursions = actions[selector].float().sum(dim=-1).mean()
                effective_depth = 2 + model.config.recursion_block_layers * recursions
                latest_train.update(
                    {
                        f"budget_{budget}_recursions": recursions.item(),
                        f"budget_{budget}_depth": effective_depth.item(),
                        f"budget_{budget}_ce": sequence_ce[selector].mean().item(),
                        f"budget_{budget}_reward": rewards[selector].mean().item(),
                        f"budget_{budget}_flops_vs_full_dense": compute[selector].mean().item(),
                    }
                )

            if step % args.log_interval == 0 or step == args.max_steps - 1:
                logger.log(latest_train, step)
                depths = ", ".join(
                    f"R{budget}:{latest_train[f'budget_{budget}_depth']:.2f}"
                    for budget in budgets
                )
                print(
                    f"step {step:5d} | reward {latest_train['mean_reward']:.4f}±"
                    f"{latest_train['reward_std']:.4f} | ce {latest_train['mean_ce_component']:.4f} "
                    f"| depths [{depths}] | kl {latest_train['mean_kl']:.5f} "
                    f"| clip {latest_train['clip_fraction']:.3f}"
                )

            if (step + 1) % args.eval_interval == 0 or step == args.max_steps - 1:
                latest_eval = evaluate_model(
                    model, dataset, device, args.batch_size, args.eval_iters,
                    routing_mode="greedy",
                )
                latest_eval = add_flops(latest_eval, model)
                validation_row = {"split": "validation", **latest_eval}
                for recursion_index, value in enumerate(latest_eval["recursion_utilization"]):
                    validation_row[f"recursion_{recursion_index + 1}_utilization"] = value
                for recursion_index, value in enumerate(
                    latest_eval["recursion_soft_utilization"]
                ):
                    validation_row[
                        f"recursion_{recursion_index + 1}_soft_utilization"
                    ] = value
                logger.log(validation_row, step)
                score = (
                    latest_eval["val_loss"]
                    + args.lambda_compute_grpo * latest_eval["estimated_flops_vs_full_dense"]
                )
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
                    f"FLOPs {latest_eval['estimated_flops_vs_full_dense']:.3f}x full dense"
                )
    finally:
        logger.close()

    training_seconds = time.perf_counter() - training_started
    selected_checkpoint = experiment_dir / "checkpoints" / "best_quality_compute.pt"
    summary = {
        "model": "mor",
        "router_type": "expert_linear",
        "training_method": "grpo",
        "grpo_variant": "mor_budget_guided_per_decision",
        "paper_reproduction": True,
        "mor_reproduction": True,
        "experiment_family": "mor_comparison",
        "seed": args.seed,
        "n_layers": model.config.n_layers,
        "recursion_steps": model.config.recursion_steps,
        "recursion_block_layers": model.config.recursion_block_layers,
        "mor_capacity_factors": model.config.mor_capacity_factors,
        "lambda_grpo": args.lambda_compute_grpo,
        "beta_kl": args.beta_kl,
        "recursion_budgets": budgets,
        "exploration_epsilon": args.exploration_epsilon,
        "ppo_clip_epsilon": args.clip_epsilon,
        "parameter_count": model.parameter_count(),
        "active_parameter_estimate": model.parameter_count(),
        "training_time_sec": training_seconds,
        "tokens_per_sec": (
            tokens_processed
            - start_step * args.batch_size * group_size * model.config.context_length
        ) / max(training_seconds, 1e-9),
        "checkpoint": str(selected_checkpoint),
        "before_grpo": before,
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
    print(f"completed MoR GRPO experiment; artifacts saved in {experiment_dir}")


if __name__ == "__main__":
    main()
