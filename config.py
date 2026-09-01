"""Serializable configurations for sparse-depth language-model experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int = 128
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    model_type: str = "sparse"
    router_type: str = "gru"
    router_dim: int = 32
    gumbel_temperature: float = 1.0
    initial_execute_probability: float = 0.9
    router_input_norm: bool = False
    router_bias: bool = True
    paper_reproduction: bool = False
    sparse_inference: bool = False
    tie_weights: bool = True
    mor_reproduction: bool = False
    recursion_steps: int = 3
    recursion_block_layers: int = 2
    mor_capacity_factors: tuple[float, ...] = (1.0, 2 / 3, 1 / 3)
    mor_router_alpha: float = 0.1
    mor_aux_loss_coefficient: float = 0.001
    mor_capacity_warmup_steps: int = 0
    mor_recursion_wise_kv: bool = True

    def __post_init__(self) -> None:
        if self.model_type == "dynamic":
            self.model_type = "sparse"
        if self.router_type == "mlp":
            self.router_type = "linear"
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.model_type not in {"dense", "sparse", "mor", "mor_skip"}:
            raise ValueError("model_type must be dense, sparse, mor, or mor_skip")
        if self.router_type not in {"linear", "gru"}:
            raise ValueError("router_type must be linear or gru")
        if not 0 < self.initial_execute_probability < 1:
            raise ValueError("initial_execute_probability must be in (0, 1)")
        self.mor_capacity_factors = tuple(float(value) for value in self.mor_capacity_factors)
        if self.model_type in {"mor", "mor_skip"}:
            expected_layers = 2 + self.recursion_steps * self.recursion_block_layers
            if self.n_layers != expected_layers:
                raise ValueError(
                    "MoR effective n_layers must equal 2 + recursion_steps * "
                    f"recursion_block_layers ({expected_layers})"
                )
            if len(self.mor_capacity_factors) != self.recursion_steps:
                raise ValueError("MoR needs one capacity factor per recursion step")
            if self.mor_capacity_factors[0] != 1.0:
                raise ValueError("Every token must traverse the first MoR recursion")
            if any(
                not 0 < value <= 1 for value in self.mor_capacity_factors
            ) or any(
                later > earlier
                for earlier, later in zip(
                    self.mor_capacity_factors, self.mor_capacity_factors[1:]
                )
            ):
                raise ValueError("MoR capacities must be positive and non-increasing")
            if not 0 < self.mor_router_alpha <= 1:
                raise ValueError("mor_router_alpha must be in (0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ModelConfig":
        values = dict(values)
        values.pop("gate_threshold", None)
        bias = values.pop("gate_bias", None)
        if bias is not None and "initial_execute_probability" not in values:
            import math

            values["initial_execute_probability"] = 1 / (1 + math.exp(-bias))
        return cls(**values)


@dataclass
class TrainConfig:
    batch_size: int = 32
    max_steps: int = 5000
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 20
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    lambda_density: float = 0.1
    target_density: float = 0.5
    density_warmup_start: float = 0.10
    density_warmup_end: float = 0.30
    quality_compute_alpha: float = 0.1
    optimizer_name: str = "adamw"
    lr_schedule: str = "cosine"
    constant_lr_steps: int = 10_000
    density_reduction: str = "mean"
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0 <= self.target_density <= 1:
            raise ValueError("target_density must be in [0, 1]")
        if not 0 <= self.density_warmup_start <= self.density_warmup_end <= 1:
            raise ValueError("density warmup must satisfy 0 <= start <= end <= 1")
        if self.optimizer_name not in {"adamw", "adafactor"}:
            raise ValueError("optimizer_name must be adamw or adafactor")
        if self.lr_schedule not in {"cosine", "paper_inverse_sqrt"}:
            raise ValueError("lr_schedule must be cosine or paper_inverse_sqrt")
        if self.density_reduction not in {"mean", "sum"}:
            raise ValueError("density_reduction must be mean or sum")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GRPOConfig:
    max_steps: int = 1000
    batch_size: int = 8
    group_size: int = 4
    policy_epochs: int = 2
    learning_rate: float = 1e-4
    transformer_lr_scale: float = 0.1
    lambda_compute_grpo: float = 0.1
    beta_kl: float = 0.01
    clip_epsilon: float = 0.2
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_iters: int = 20
    log_interval: int = 10
    router_only: bool = True
    kl_in_reward: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2")
        if self.policy_epochs < 1:
            raise ValueError("policy_epochs must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
