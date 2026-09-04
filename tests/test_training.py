from __future__ import annotations

import torch

from tiny_deepseek.core.losses import density_loss, scheduled_coefficient
from tiny_deepseek.data.math import MathExample
from tiny_deepseek.training.math_grpo import curriculum_candidates
from tiny_deepseek.training.math_sft import curriculum_source


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


def test_math_curriculum_moves_to_gsm_heavy_batches() -> None:
    assert curriculum_source(0, 900, 42) == "synthetic_easy"
    assert curriculum_source(400, 900, 42) == "synthetic_medium"
    assert curriculum_source(800, 900, 42) == "synthetic_hard"
    sources = [curriculum_source(10_000 + step, 1_000, 42) for step in range(1_000)]
    gsm_fraction = sources.count("gsm_train") / len(sources)
    assert 0.85 <= gsm_fraction <= 0.95


def test_grpo_curriculum_admits_harder_prompts_progressively() -> None:
    pool = [
        MathExample("easy", "work", "1", "easy", "addition"),
        MathExample("medium", "work", "2", "medium", "mixed"),
        MathExample("hard", "work", "3", "hard", "mixed"),
    ]
    assert [item.difficulty for item in curriculum_candidates(pool, 0, 90)] == ["easy"]
    assert {
        item.difficulty for item in curriculum_candidates(pool, 40, 90)
    } == {"easy", "medium"}
    assert {
        item.difficulty for item in curriculum_candidates(pool, 80, 90)
    } == {"easy", "medium", "hard"}
