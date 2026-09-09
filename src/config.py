from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class ExperimentConfig:
    """Single source of truth for a lab-reproducible run."""

    data_dir: str = "data"
    artifact_dir: str = "artifacts"

    emb_dim: int = 32
    hidden_dim: int = 32
    cold_threshold: int = 10
    cold_ratio: float | None = None

    epochs: int = 150
    batch_size: int = 256
    lr: float = 0.005
    tau: float = 0.1
    l2_reg: float = 5e-4
    patience: int = 20
    val_warm_frac: float = 0.1

    n_test_users: int = 1000
    seed: int = 42
    n_seeds: int = 1

    use_title: bool = True
    title_svd_dim: int = 16
    protocol: str = "academic"
    ks: List[int] = field(default_factory=lambda: [5, 10, 20])
    n_bootstrap: int = 200

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_args(cls, args) -> "ExperimentConfig":
        values = {}
        for name in cls.__dataclass_fields__:
            if hasattr(args, name):
                values[name] = getattr(args, name)
        return cls(**values)
