"""Budget-guided GRPO for inner SkipLayer gates inside a frozen MoR model."""

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
from model import MixtureOfRecursionsSkipLayerTransformer
from train_mor_skip import add_hybrid_flops, add_router_fields, metric_fields
from utils import (
    estimate_dense_block_flops,
    gradient_norm,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
    select_device,
    set_seed,
    synchronize_device,
    write_json,
)


GRPO_FIELDS = [
    "step", "split", "mean_reward", "reward_std", "mean_ce_component",
    "mean_compute_penalty", "mean_kl", "policy_loss", "total_policy_loss",
    "mean_advantage", "advantage_std", "routing_entropy", "clip_fraction",
    "mean_probability_ratio", "learning_rate", "gradient_norm",
    "skip_router_gradient_norm", "frozen_parameter_gradient_norm",
    "tokens_processed", "seconds_per_step", "tokens_per_second",
]


def masked_policy_kl(
    current_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    decision_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_log = F.log_softmax(current_logits, dim=-1)
    reference_log = F.log_softmax(reference_logits, dim=-1)
    probability = current_log.exp()
    decision_kl = (probability * (current_log - reference_log)).sum(dim=-1)
    valid = decision_mask.float()
    per_trajectory = (decision_kl * valid).sum(dim=(1, 2)) / valid.sum(
        dim=(1, 2)
    ).clamp_min(1)
    return per_trajectory, decision_kl[decision_mask].mean()


def trajectory_hybrid_flop_fraction(
    model: MixtureOfRecursionsSkipLayerTransformer,
    mor_actions: torch.Tensor,
    skip_actions: torch.Tensor,
) -> torch.Tensor:
    """Exact paper-style estimated FLOP fraction for every sampled trajectory."""
    config = model.config
    length = float(config.context_length)
    d, ff = config.d_model, config.d_ff
    full_layer = 8 * length * d * d + 4 * length * length * d + 4 * length * d * ff
    total = mor_actions.new_full((mor_actions.shape[0],), 2 * full_layer, dtype=torch.float32)
    previous = torch.ones_like(total)
    inner = 0
    for recursion in range(config.recursion_steps):
        admitted = mor_actions[..., recursion].float().mean(dim=1).minimum(previous)
        total = total + 2 * previous * length * d
        for _ in range(config.recursion_block_layers):
            executed = skip_actions[..., inner].float().mean(dim=1).minimum(admitted)
            total = total + (
                4 * admitted * length * d * d
                + 4 * executed * length * d * d
                + 4 * executed * admitted * length * length * d
                + 4 * executed * length * d * ff
                + 4 * admitted * length * d
            )
            inner += 1
        previous = admitted
    dense = estimate_dense_block_flops(config, config.context_length)
    return total / dense


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", help="Supervised MoR+Skip checkpoint")
    result.add_argument("--resume", help="Resume a MoR+Skip GRPO checkpoint")
    result.add_argument("--experiment-dir")
    result.add_argument("--data-path", default="data/input.txt")
    result.add_argument("--device", default="auto")
    result.add_argument("--max-steps", type=int, default=300)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument(
        "--skip-density-budgets", type=float, nargs="+",
        default=[0.25, 0.5, 0.75, 1.0],
    )
    result.add_argument("--exploration-epsilon", type=float, default=0.8)
    result.add_argument("--policy-epochs", type=int, default=2)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--lambda-compute-grpo", type=float, default=1.0)
    result.add_argument("--beta-kl", type=float, default=0.01)
    result.add_argument("--clip-epsilon", type=float, default=0.5)
    result.add_argument("--grad-clip", type=float, default=1.0)
    result.add_argument("--eval-interval", type=int, default=50)
    result.add_argument("--eval-iters", type=int, default=20)
    result.add_argument("--log-interval", type=int, default=10)
    result.add_argument("--seed", type=int, default=42)
    return result


def budget_label(value: float) -> str:
    return f"P{int(round(100 * value)):03d}"


def main() -> None:
    args = parser().parse_args()
    if not args.checkpoint and not args.resume:
        raise SystemExit("Provide --checkpoint for a new run or --resume")
    budgets = [float(value) for value in args.skip_density_budgets]
    if budgets != sorted(set(budgets)) or budgets[-1] != 1.0:
        raise ValueError("skip-density-budgets must be unique, sorted, and end at 1.0")
    if any(value < 0 or value > 1 for value in budgets):
        raise ValueError("skip density budgets must lie in [0,1]")
    if not 0 <= args.exploration_epsilon < 1:
        raise ValueError("exploration-epsilon must be in [0,1); anchor uses 1 internally")

    device = select_device(args.device)
    set_seed(args.seed)
    source = args.resume or args.checkpoint
    model, checkpoint = load_checkpoint(source, device)
    if not isinstance(model, MixtureOfRecursionsSkipLayerTransformer):
        raise ValueError("GRPO requires a supervised MoR+Skip checkpoint")
    dataset = TinyShakespeareData(args.data_path, model.config.context_length)
    if dataset.stoi != checkpoint["stoi"]:
        raise ValueError("Checkpoint vocabulary differs from the dataset")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    skip_parameters = list(model.skip_router_parameters())
    for parameter in skip_parameters:
        parameter.requires_grad_(True)
    frozen_parameters = [p for p in model.parameters() if not p.requires_grad]
    reference_router = copy.deepcopy(model.skip_router).to(device).eval()
    for parameter in reference_router.parameters():
        parameter.requires_grad_(False)
    if args.resume and checkpoint.get("reference_router_state"):
        reference_router.load_state_dict(checkpoint["reference_router_state"])

    optimizer = torch.optim.AdamW(skip_parameters, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.max_steps, 1), eta_min=args.learning_rate * 0.1
    )
    start_step = 0
    best = {"val_loss": float("inf"), "quality_compute_score": float("inf")}
    if args.resume:
        start_step = restore_training_state(checkpoint, optimizer, scheduler)
        best.update(checkpoint.get("best_metrics") or {})

    supervised_dir = Path(args.checkpoint).parent.parent if args.checkpoint else Path(args.resume).parent.parent
    experiment_dir = Path(
        args.experiment_dir
        or f"experiments/{supervised_dir.name}_grpo_lam{args.lambda_compute_grpo:g}_seed{args.seed}"
    )
    for child in ("checkpoints", "plots", "samples", "routing_visualizations"):
        (experiment_dir / child).mkdir(parents=True, exist_ok=True)

    before = add_hybrid_flops(
        evaluate_model(model, dataset, device, args.batch_size, args.eval_iters), model
    )
    full_config = {
        "stage": "mor_inner_skip_grpo",
        "model": model.config.to_dict(),
        "source_supervised_checkpoint": args.checkpoint,
        "training": vars(args),
        "before_grpo": before,
        "frozen_backbone": True,
        "frozen_mor_router": True,
        "sampling_note": (
            "MoR admission is deterministic greedy. Inner SkipLayer actions use an exact "
            "epsilon mixture of the policy and a requested conditional execution density. "
            "The 1.0 execute-all-eligible anchor is excluded from policy loss."
        ),
    }
    write_json(experiment_dir / "config.json", full_config)

    budget_fields = [
        f"budget_{budget_label(budget)}_{metric}"
        for budget in budgets
        for metric in (
            "conditional_skip_density", "effective_depth", "ce", "reward",
            "flops_vs_full_dense",
        )
    ]
    fields = list(dict.fromkeys(GRPO_FIELDS + metric_fields(model) + budget_fields))
    logger = StructuredLogger(
        experiment_dir, fields, purge_step=start_step if args.resume else None
    )
    group_size = len(budgets)
    latest_eval = before
    latest_train: dict[str, Any] = {}
    tokens_processed = (
        start_step * args.batch_size * group_size * model.config.context_length
    )
    measured_seconds = 0.0
    training_started = time.perf_counter()
    try:
        for step in range(start_step, args.max_steps):
            model.eval()
            synchronize_device(device)
            started = time.perf_counter()
            base_x, base_y = dataset.get_batch("train", args.batch_size, device)
            x = base_x.repeat_interleave(group_size, dim=0)
            y = base_y.repeat_interleave(group_size, dim=0)
            trajectory_budgets = torch.tensor(
                budgets, device=device, dtype=torch.float32
            ).repeat(args.batch_size)
            epsilons = torch.full_like(trajectory_budgets, args.exploration_epsilon)
            anchor_mask = trajectory_budgets.eq(1.0)
            epsilons[anchor_mask] = 1.0
            trainable_mask = ~anchor_mask

            with torch.no_grad():
                sampled = model(
                    x, y, routing_mode="budget",
                    target_skip_densities=trajectory_budgets,
                    exploration_epsilon=epsilons,
                )
                reference = model(
                    x, y, routing_mode="greedy", actions=sampled.actions,
                    router_override=reference_router,
                )
                sequence_ce = sampled.token_losses.mean(dim=1)
                compute = trajectory_hybrid_flop_fraction(
                    model, sampled.mor_actions, sampled.actions
                )
                sampled_kl, _ = masked_policy_kl(
                    sampled.route_logits,
                    reference.route_logits,
                    sampled.routing_decision_mask,
                )
                rewards = grpo_reward(
                    sequence_ce, compute, sampled_kl,
                    args.lambda_compute_grpo, 0.0,
                )
                reward_groups = rewards.view(args.batch_size, group_size)
                advantages = group_relative_advantages(reward_groups).reshape(-1)
                behavior = sampled.behavior_log_probs.detach()
                actions = sampled.actions.detach()

            losses, kls, entropies, ratios, clips = [], [], [], [], []
            for _ in range(args.policy_epochs):
                current = model(x, y, routing_mode="greedy", actions=actions)
                with torch.no_grad():
                    reference = model(
                        x, y, routing_mode="greedy", actions=actions,
                        router_override=reference_router,
                    )
                policy_loss, ratio, clip = clipped_grpo_loss_per_decision(
                    current.action_log_probs,
                    behavior,
                    advantages,
                    args.clip_epsilon,
                    trajectory_mask=trainable_mask,
                    decision_mask=current.routing_decision_mask,
                )
                _, kl = masked_policy_kl(
                    current.route_logits,
                    reference.route_logits,
                    current.routing_decision_mask,
                )
                total = policy_loss + args.beta_kl * kl
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                skip_grad = gradient_norm(skip_parameters)
                frozen_grad = gradient_norm(frozen_parameters)
                torch.nn.utils.clip_grad_norm_(skip_parameters, args.grad_clip)
                optimizer.step()
                losses.append(policy_loss.detach())
                kls.append(kl.detach())
                valid_entropy = current.routing_entropy[current.routing_decision_mask]
                entropies.append(valid_entropy.mean().detach())
                ratios.append(ratio.detach())
                clips.append(clip.detach())
            scheduler.step()
            synchronize_device(device)
            seconds = time.perf_counter() - started
            measured_seconds += seconds
            tokens_processed += x.numel()

            latest_train = {
                "split": "grpo_train",
                "mean_reward": rewards.mean().item(),
                "reward_std": rewards.std(unbiased=False).item(),
                "mean_ce_component": sequence_ce.mean().item(),
                "mean_compute_penalty": (
                    args.lambda_compute_grpo * compute
                ).mean().item(),
                "mean_kl": torch.stack(kls).mean().item(),
                "policy_loss": torch.stack(losses).mean().item(),
                "total_policy_loss": (
                    torch.stack(losses).mean() + args.beta_kl * torch.stack(kls).mean()
                ).item(),
                "mean_advantage": advantages.mean().item(),
                "advantage_std": advantages.std(unbiased=False).item(),
                "routing_entropy": torch.stack(entropies).mean().item(),
                "clip_fraction": torch.stack(clips).mean().item(),
                "mean_probability_ratio": torch.stack(ratios).mean().item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": skip_grad,
                "skip_router_gradient_norm": skip_grad,
                "frozen_parameter_gradient_norm": frozen_grad,
                "tokens_processed": tokens_processed,
                "seconds_per_step": seconds,
                "tokens_per_second": x.numel() / max(seconds, 1e-9),
            }
            sampled_metrics = {
                "recursion_utilization": sampled.recursion_utilization.tolist(),
                "recursion_soft_utilization": sampled.recursion_soft_utilization.tolist(),
                "skip_conditional_utilization": sampled.skip_conditional_utilization.tolist(),
                "skip_soft_conditional_utilization": sampled.skip_soft_conditional_utilization.tolist(),
                "combined_block_utilization": sampled.combined_block_utilization.tolist(),
            }
            add_router_fields(latest_train, sampled_metrics, model)
            for group_index, budget in enumerate(budgets):
                selector = torch.arange(group_index, x.shape[0], group_size, device=device)
                valid = sampled.routing_decision_mask[selector]
                conditional_density = (
                    sampled.skip_hard_gates[selector][valid].float().mean()
                )
                label = budget_label(budget)
                latest_train.update(
                    {
                        f"budget_{label}_conditional_skip_density": conditional_density.item(),
                        f"budget_{label}_effective_depth": sampled.hard_gates[selector].float().sum(dim=-1).mean().item(),
                        f"budget_{label}_ce": sequence_ce[selector].mean().item(),
                        f"budget_{label}_reward": rewards[selector].mean().item(),
                        f"budget_{label}_flops_vs_full_dense": compute[selector].mean().item(),
                    }
                )

            if step % args.log_interval == 0 or step == args.max_steps - 1:
                logger.log(latest_train, step)
                depths = ", ".join(
                    f"{budget_label(b)}:{latest_train[f'budget_{budget_label(b)}_effective_depth']:.2f}"
                    for b in budgets
                )
                print(
                    f"step {step:5d} | reward {latest_train['mean_reward']:.4f} | "
                    f"depths [{depths}] | clip {latest_train['clip_fraction']:.3f}"
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
                score = latest_eval["val_loss"] + args.lambda_compute_grpo * latest_eval[
                    "estimated_flops_vs_full_dense"
                ]
                checkpoint_kwargs = dict(
                    model=model, optimizer=optimizer, scheduler=scheduler, step=step + 1,
                    stoi=dataset.stoi, itos=dataset.itos, training_config=full_config,
                    best_metrics=best, reference_router_state=reference_router.state_dict(),
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
        "training_method": "grpo",
        "grpo_variant": "mor_inner_skiplayer_budget_guided_per_decision",
        "experiment_family": "mor_skip_hybrid",
        "paper_reproduction": True,
        "mor_reproduction": True,
        "seed": args.seed,
        "lambda_grpo": args.lambda_compute_grpo,
        "beta_kl": args.beta_kl,
        "skip_density_budgets": budgets,
        "exploration_epsilon": args.exploration_epsilon,
        "ppo_clip_epsilon": args.clip_epsilon,
        "parameter_count": model.parameter_count(),
        "active_parameter_estimate": model.parameter_count(),
        "training_time_sec": training_seconds,
        "seconds_per_step": measured_seconds / max(args.max_steps - start_step, 1),
        "tokens_per_sec": (
            (tokens_processed - start_step * args.batch_size * group_size * model.config.context_length)
            / max(training_seconds, 1e-9)
        ),
        "checkpoint": str(selected),
        "before_grpo": before,
        **latest_eval,
        **latest_train,
    }
    write_json(experiment_dir / "summary.json", summary)
    save_checkpoint(
        path=experiment_dir / "checkpoints/latest.pt", model=model,
        optimizer=optimizer, scheduler=scheduler, step=args.max_steps,
        stoi=dataset.stoi, itos=dataset.itos, training_config=full_config,
        best_metrics=best, summary=summary,
        reference_router_state=reference_router.state_dict(),
    )
    print(f"completed MoR+Skip GRPO experiment: {experiment_dir}")


if __name__ == "__main__":
    main()
