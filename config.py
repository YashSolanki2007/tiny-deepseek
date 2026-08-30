"""Configuration objects shared by training and inference."""

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
    model_type: str = "dynamic"  # "dense" or "dynamic"
    router_type: str = "gru"  # "gru" or "mlp"
    router_dim: int = 32
    gate_threshold: float = 0.5
    gate_bias: float = 2.2
    tie_weights: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.model_type not in {"dense", "dynamic"}:
            raise ValueError("model_type must be 'dense' or 'dynamic'")
        if self.router_type not in {"gru", "mlp"}:
            raise ValueError("router_type must be 'gru' or 'mlp'")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ModelConfig":
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
    lambda_compute: float = 0.01
    compute_loss: str = "linear"  # "linear" or "target"
    target_compute: float = 0.5
    compute_warmup_start: float = 0.10
    compute_warmup_end: float = 0.30
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.compute_loss not in {"linear", "target"}:
            raise ValueError("compute_loss must be 'linear' or 'target'")
        if not 0 <= self.compute_warmup_start <= self.compute_warmup_end <= 1:
            raise ValueError("compute warmup fractions must satisfy 0 <= start <= end <= 1")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
