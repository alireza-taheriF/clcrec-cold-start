"""Centralized configuration: hyperparameters and filesystem paths.

Everything the pipeline needs to be reproducible lives here. Import `Config`
and either use the defaults or override fields when constructing it. The CLI in
`main.py` maps command-line flags onto these fields.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from typing import List


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Anchor every path to the repository root so the code runs the same regardless
# of the current working directory.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Paths:
    root: str = ROOT_DIR
    data_dir: str = os.path.join(ROOT_DIR, "data")
    results_dir: str = os.path.join(ROOT_DIR, "results")
    artifacts_dir: str = os.path.join(ROOT_DIR, "results", "artifacts")

    def ensure(self) -> "Paths":
        """Create output directories if they do not exist yet."""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        return self

    # Convenience accessors for well-known artifact files.
    @property
    def item_embeddings(self) -> str:
        return os.path.join(self.artifacts_dir, "item_embeddings.npz")

    @property
    def metadata(self) -> str:
        return os.path.join(self.artifacts_dir, "metadata.json")

    @property
    def metrics_json(self) -> str:
        return os.path.join(self.results_dir, "metrics.json")

    @property
    def metrics_summary(self) -> str:
        return os.path.join(self.results_dir, "metrics_summary.md")

    @property
    def results_png(self) -> str:
        return os.path.join(self.results_dir, "results.png")


@dataclass
class Config:
    """Hyperparameters shared by both the NumPy and PyTorch backends.

    Fields are grouped by concern: data prep, model, training, evaluation.
    """

    # -- data preparation -------------------------------------------------- #
    pos_threshold: int = 4       # ratings >= this count as a positive interaction
    cold_threshold: int = 10     # items with fewer positives than this are "cold"
    emb_dim: int = 32            # SVD rank and encoder output dimension
    use_text_features: bool = True  # Innovation 1: enrich with title/semantic TF-IDF+SVD
    text_dim: int = 32           # dimension of text semantic features

    # -- model & innovations ----------------------------------------------- #
    hidden_dim: int = 32         # NumPy encoder hidden width
    torch_hidden_dim: int = 64   # PyTorch encoder hidden width (a bit more capacity)
    tau: float = 0.1             # InfoNCE temperature
    learnable_tau: bool = False  # torch backend only: learn tau as a parameter
    user_attention: bool = False # Innovation 2: attention-based user profiling
    hard_negatives: bool = False # Innovation 3: content-aware hard negative mining
    hard_negative_weight: float = 0.5 # weight for hard negatives in InfoNCE
    joint_training: bool = False # Innovation 5: joint BPR + InfoNCE end-to-end
    bpr_weight: float = 1.0      # weight for BPR collaborative loss in joint training

    # -- training ---------------------------------------------------------- #
    epochs: int = 150
    batch_size: int = 256
    lr: float = 0.005
    l2_reg: float = 5e-4         # weight decay / L2 regularization
    seed: int = 42

    # -- evaluation -------------------------------------------------------- #
    n_test_users: int = 1000
    ks: List[int] = field(default_factory=lambda: [5, 10, 20])

    # -- runtime ----------------------------------------------------------- #
    backend: str = "numpy"       # "numpy" or "torch"
    device: str = "auto"         # torch backend: "auto" | "cpu" | "cuda"

    paths: Paths = field(default_factory=Paths)

    def to_dict(self) -> dict:
        """Serializable view of the config (paths flattened to strings)."""
        d = asdict(self)
        return d


DEFAULT_CONFIG = Config()
