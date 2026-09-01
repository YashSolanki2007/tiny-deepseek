from __future__ import annotations

import torch

from losses import (
    clipped_grpo_loss,
    clipped_grpo_loss_per_decision,
    grpo_reward,
    group_relative_advantages,
    masked_density_loss,
)


def test_grpo_reward_combines_ce_compute_and_kl() -> None:
    ce = torch.tensor([2.0, 3.0])
    compute = torch.tensor([0.5, 0.25])
    kl = torch.tensor([0.1, 0.2])
    reward = grpo_reward(ce, compute, kl, lambda_compute=0.2, beta_kl=0.01)
    torch.testing.assert_close(reward, -ce - 0.2 * compute - 0.01 * kl)


def test_group_relative_advantages_are_normalized_per_group() -> None:
    advantages = group_relative_advantages(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    torch.testing.assert_close(advantages.mean(dim=1), torch.zeros(1), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        advantages.std(dim=1, unbiased=False), torch.ones(1), atol=1e-5, rtol=0
    )


def test_clipped_objective_reports_clipping() -> None:
    old = torch.zeros(2)
    new = torch.log(torch.tensor([2.0, 0.5]))
    loss, ratio, clip_fraction = clipped_grpo_loss(
        new, old, torch.tensor([1.0, -1.0]), 0.2
    )
    assert torch.isfinite(loss)
    assert ratio.item() == 1.25
    assert clip_fraction.item() == 1.0


def test_per_decision_objective_excludes_anchor_trajectory() -> None:
    behavior = torch.zeros(2, 3, 4)
    current = behavior.clone()
    current[1] = 10.0
    loss, ratio, clip_fraction = clipped_grpo_loss_per_decision(
        current,
        behavior,
        torch.tensor([1.0, -1.0]),
        clip_epsilon=0.2,
        trajectory_mask=torch.tensor([True, False]),
    )
    torch.testing.assert_close(loss, torch.tensor(-1.0))
    torch.testing.assert_close(ratio, torch.tensor(1.0))
    torch.testing.assert_close(clip_fraction, torch.tensor(0.0))


def test_per_decision_objective_honors_hierarchical_decision_mask() -> None:
    behavior = torch.zeros(1, 2, 3)
    current = behavior.clone()
    current[0, 1, 2] = 10.0
    decision_mask = torch.tensor([[[False, True, False], [False, True, False]]])
    loss, ratio, clipped = clipped_grpo_loss_per_decision(
        current,
        behavior,
        torch.tensor([1.0]),
        clip_epsilon=0.2,
        decision_mask=decision_mask,
    )
    torch.testing.assert_close(loss, torch.tensor(-1.0))
    torch.testing.assert_close(ratio, torch.tensor(1.0))
    torch.testing.assert_close(clipped, torch.tensor(0.0))


def test_masked_density_ignores_tokens_not_admitted_by_mor() -> None:
    gates = torch.tensor(
        [[[1.0, 0.0], [0.0, 0.0]]], requires_grad=True
    )
    eligible = torch.tensor([[[True, True], [False, False]]])
    loss = masked_density_loss(gates, eligible, target_density=0.5)
    torch.testing.assert_close(loss, torch.tensor(0.25))
    loss.backward()
    assert gates.grad[0, 1].eq(0).all()
