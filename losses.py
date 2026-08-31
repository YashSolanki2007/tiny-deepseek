"""Supervised density and GRPO objectives."""

from __future__ import annotations

import torch


def density_loss(
    hard_gates: torch.Tensor,
    target_density: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """Squared per-layer capacity error using hard-forward/soft-backward gates.

    The SkipLayer paper sums the error over layers. ``mean`` remains available
    for backward compatibility with the earlier exploratory experiments.
    """
    per_layer = (hard_gates.mean(dim=(0, 1)) - target_density).square()
    if reduction == "sum":
        return per_layer.sum()
    if reduction == "mean":
        return per_layer.mean()
    raise ValueError("density reduction must be 'mean' or 'sum'")


def scheduled_coefficient(
    step: int, total_steps: int, target: float, start: float = 0.1, end: float = 0.3
) -> float:
    progress = step / max(total_steps, 1)
    if progress < start:
        return 0.0
    if progress < end:
        return target * (progress - start) / max(end - start, 1e-12)
    return target


def grpo_reward(
    sequence_ce: torch.Tensor,
    compute_fraction: torch.Tensor,
    kl: torch.Tensor,
    lambda_compute: float,
    beta_kl: float,
) -> torch.Tensor:
    return -sequence_ce - lambda_compute * compute_fraction - beta_kl * kl


def group_relative_advantages(rewards: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    return centered / (rewards.std(dim=1, keepdim=True, unbiased=False) + epsilon)


def clipped_grpo_loss(
    new_log_probability: torch.Tensor,
    old_log_probability: torch.Tensor,
    advantage: torch.Tensor,
    clip_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ratio = torch.exp(new_log_probability - old_log_probability)
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    surrogate = torch.minimum(ratio * advantage, clipped * advantage)
    return -surrogate.mean(), ratio.mean(), (ratio.ne(clipped)).float().mean()


def clipped_grpo_loss_per_decision(
    new_log_probability: torch.Tensor,
    behavior_log_probability: torch.Tensor,
    advantage: torch.Tensor,
    clip_epsilon: float,
    trajectory_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clipped GRPO/PPO objective over token-layer routing decisions.

    A sequence-level group-relative advantage is broadcast over that
    trajectory's decisions. ``behavior_log_probability`` is the probability
    under the distribution that actually sampled each action, which permits
    budget-guided exploration without pretending those actions were on-policy.
    """
    if new_log_probability.shape != behavior_log_probability.shape:
        raise ValueError("new and behavior log-probability tensors must match")
    if advantage.shape != (new_log_probability.shape[0],):
        raise ValueError("advantage must have shape [trajectories]")
    ratio = torch.exp(new_log_probability - behavior_log_probability)
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    expanded_advantage = advantage[:, None, None].expand_as(ratio)
    surrogate = torch.minimum(ratio * expanded_advantage, clipped * expanded_advantage)
    if trajectory_mask is not None:
        if trajectory_mask.shape != (new_log_probability.shape[0],):
            raise ValueError("trajectory_mask must have shape [trajectories]")
        valid = trajectory_mask[:, None, None].expand_as(ratio)
        if not bool(valid.any()):
            raise ValueError("trajectory_mask excludes every trajectory")
        surrogate = surrogate[valid]
        ratio = ratio[valid]
        clipped = clipped[valid]
    return -surrogate.mean(), ratio.mean(), ratio.ne(clipped).float().mean()
