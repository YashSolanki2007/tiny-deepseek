from __future__ import annotations

import torch

from losses import density_loss, scheduled_coefficient


def test_density_schedule() -> None:
    assert scheduled_coefficient(0, 100, 0.1, 0.1, 0.3) == 0
    assert scheduled_coefficient(10, 100, 0.1, 0.1, 0.3) == 0
    assert abs(scheduled_coefficient(20, 100, 0.1, 0.1, 0.3) - 0.05) < 1e-8
    assert scheduled_coefficient(30, 100, 0.1, 0.1, 0.3) == 0.1


def test_density_objective_is_zero_at_layerwise_target() -> None:
    gates = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]
    )
    torch.testing.assert_close(density_loss(gates, 0.5), torch.tensor(0.0))


def test_paper_density_objective_sums_over_layers() -> None:
    gates = torch.zeros(2, 3, 4)
    mean_loss = density_loss(gates, 0.5, reduction="mean")
    paper_loss = density_loss(gates, 0.5, reduction="sum")
    torch.testing.assert_close(paper_loss, 4 * mean_loss)
