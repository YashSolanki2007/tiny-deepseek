"""Optimizer variants needed by the paper-reproduction protocol."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


class FixedDecayAdafactor(torch.optim.Optimizer):
    """Adafactor with the fixed beta2 used by the SkipLayer paper.

    The paper reports beta1=0 and beta2=0.99. With beta1=0 there is no
    first-moment accumulator. Matrix-like tensors use factored second moments;
    vectors and scalars use an unfactored accumulator.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.1,
        beta2: float = 0.99,
        eps: tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= beta2 < 1:
            raise ValueError("beta2 must be in [0, 1)")
        defaults = dict(
            lr=lr,
            beta2=beta2,
            eps=eps,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float().square().mean().sqrt()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta2 = group["beta2"]
            eps1, eps2 = group["eps"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("FixedDecayAdafactor does not support sparse gradients")
                state = self.state[parameter]
                state["step"] = state.get("step", 0) + 1
                grad_float = gradient.float()
                grad_squared = grad_float.square().add_(eps1)
                if gradient.ndim >= 2:
                    row_shape = gradient.shape[:-1]
                    col_shape = gradient.shape[:-2] + gradient.shape[-1:]
                    if "exp_avg_sq_row" not in state:
                        state["exp_avg_sq_row"] = torch.zeros(
                            row_shape, dtype=torch.float32, device=gradient.device
                        )
                        state["exp_avg_sq_col"] = torch.zeros(
                            col_shape, dtype=torch.float32, device=gradient.device
                        )
                    row = state["exp_avg_sq_row"]
                    col = state["exp_avg_sq_col"]
                    row.mul_(beta2).add_(grad_squared.mean(dim=-1), alpha=1 - beta2)
                    col.mul_(beta2).add_(grad_squared.mean(dim=-2), alpha=1 - beta2)
                    row_factor = (row / row.mean(dim=-1, keepdim=True).clamp_min(eps1)).rsqrt()
                    update = grad_float * row_factor.unsqueeze(-1) * col.rsqrt().unsqueeze(-2)
                else:
                    if "exp_avg_sq" not in state:
                        state["exp_avg_sq"] = torch.zeros_like(
                            gradient, dtype=torch.float32
                        )
                    variance = state["exp_avg_sq"]
                    variance.mul_(beta2).add_(grad_squared, alpha=1 - beta2)
                    update = grad_float * variance.rsqrt()
                update.div_(
                    torch.clamp(
                        self._rms(update) / group["clip_threshold"], min=1.0
                    )
                )
                parameter_scale = (
                    max(group["eps"][1], self._rms(parameter).item())
                    if group["scale_parameter"]
                    else 1.0
                )
                step_size = group["lr"] * parameter_scale
                if group["weight_decay"]:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update.to(parameter.dtype), alpha=-step_size)
        return loss
