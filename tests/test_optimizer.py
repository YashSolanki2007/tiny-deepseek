from __future__ import annotations

import torch

from optimizers import FixedDecayAdafactor


def test_fixed_decay_adafactor_updates_matrix_and_vector_parameters() -> None:
    matrix = torch.nn.Parameter(torch.randn(4, 3))
    vector = torch.nn.Parameter(torch.randn(3))
    before_matrix = matrix.detach().clone()
    before_vector = vector.detach().clone()
    optimizer = FixedDecayAdafactor([matrix, vector], lr=0.1, beta2=0.99)
    (matrix.square().mean() + vector.square().mean()).backward()
    optimizer.step()
    assert torch.isfinite(matrix).all()
    assert torch.isfinite(vector).all()
    assert not torch.equal(matrix, before_matrix)
    assert not torch.equal(vector, before_vector)
    state = optimizer.state_dict()
    assert state["state"]
