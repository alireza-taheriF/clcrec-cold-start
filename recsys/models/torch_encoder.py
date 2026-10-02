"""PyTorch implementation of the content encoder.

This mirrors the from-scratch NumPy encoder in :mod:`recsys.models.numpy_encoder`
but leans on autograd. The architecture is intentionally close to the teaching
version so the two backends are directly comparable:

    content_dim -> hidden_dim -> emb_dim, ReLU in between, L2-normalized output.

The default ``hidden_dim`` here is 64 (vs. the NumPy default of 32) to give the
Torch variant a little more capacity, but everything is driven by config.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - torch is an optional backend
    raise ImportError(
        "PyTorch is required for the torch backend. Install with "
        "`pip install torch`, or use --backend numpy."
    ) from exc


def get_device(prefer: str = "auto") -> "torch.device":
    """Pick a compute device.

    ``prefer`` may be ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``. ``auto``
    prefers CUDA, then Apple MPS, then CPU.
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        return torch.device("cuda")
    if prefer == "mps":
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchContentEncoder(nn.Module):
    """A small MLP mapping content features to the collaborative embedding space.

    Parameters
    ----------
    content_dim:
        Input feature dimension (genres + year).
    emb_dim:
        Output embedding dimension; must match the SVD collaborative dim.
    hidden_dim:
        Hidden layer width.
    tau:
        InfoNCE temperature. When ``learnable_tau`` is True this is used as the
        initial value of a trainable log-temperature parameter.
    learnable_tau:
        If True, temperature is optimized jointly with the network.
    """

    def __init__(
        self,
        content_dim: int,
        emb_dim: int,
        hidden_dim: int = 64,
        tau: float = 0.1,
        learnable_tau: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(content_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )
        self.learnable_tau = learnable_tau
        if learnable_tau:
            # Parameterize log(tau) for positivity and stable gradients.
            self.log_tau = nn.Parameter(torch.tensor(float(np.log(tau))))
        else:
            self.register_buffer("_tau", torch.tensor(float(tau)))

    @property
    def tau(self) -> "torch.Tensor":
        if self.learnable_tau:
            return self.log_tau.exp()
        return self._tau

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        z = self.net(x)
        return F.normalize(z, p=2, dim=1, eps=1e-8)


def info_nce_loss(
    z_content: "torch.Tensor",
    z_collab: "torch.Tensor",
    tau: "torch.Tensor",
    hard_negatives: bool = False,
    hard_weight: float = 0.5,
    content_features: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """InfoNCE aligning content embeddings with fixed collaborative embeddings.

    Positive pair for row ``i`` is ``(z_content[i], z_collab[i])``; all other
    items in the batch are negatives. When ``hard_negatives`` is True, structurally
    or semantically similar negatives receive higher weighting to sharpen boundaries.
    """
    logits = z_content @ z_collab.t() / tau

    if hard_negatives:
        batch_size = logits.size(0)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=logits.device)
        if content_features is not None:
            norm_c = F.normalize(content_features, p=2, dim=1, eps=1e-8)
            content_sim = norm_c @ norm_c.t()
            hardness = hard_weight * (content_sim / tau)
            logits = logits + mask.float() * hardness
        else:
            neg_logits = logits[mask].detach()
            mean_neg = neg_logits.mean()
            std_neg = neg_logits.std() + 1e-6
            hardness = hard_weight * torch.clamp((logits - mean_neg) / std_neg, min=0.0)
            logits = logits + mask.float() * hardness

    targets = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, targets)


class TorchRecommender:
    """:class:`~recsys.models.base.RecommenderModel` adapter over :class:`TorchContentEncoder`.

    Mirrors :class:`~recsys.models.numpy_encoder.NumpyRecommender` so the two
    backends are interchangeable behind the shared interface used by the trainer,
    evaluator, and API.
    """

    backend = "torch"

    def __init__(self, config):
        self.config = config
        self.device = get_device(getattr(config, "device", "auto"))
        self.encoder: Optional[TorchContentEncoder] = None
        self.item_emb: Optional[np.ndarray] = None   # (n_movies, emb_dim) after fit
        self.history: List[float] = []
        self._data = None

    def fit(self, data, val_data=None) -> "TorchRecommender":
        from recsys.training.trainer_torch import train_torch
        self.encoder, self.history = train_torch(data, self.config, device=self.device)
        self.item_emb = self.encode_items(data.content_features)
        self._data = data
        return self

    def encode_items(self, item_content_matrix: np.ndarray) -> np.ndarray:
        if self.encoder is None:
            raise RuntimeError("Model must be fit before encoding items.")
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
        out = []
        bs = 512
        with torch.no_grad():
            for start in range(0, len(item_content_matrix), bs):
                chunk = item_content_matrix[start:start + bs]
                x = torch.as_tensor(np.asarray(chunk, dtype=np.float32), device=self.device)
                out.append(self.encoder(x).cpu().numpy())
        return np.vstack(out).astype(np.float32)

    def encode_user(self, liked_item_indices) -> Optional[np.ndarray]:
        idx = list(liked_item_indices)
        if not idx or self.item_emb is None:
            return None
        u = self.item_emb[idx].mean(axis=0)
        norm = np.linalg.norm(u)
        return u / norm if norm > 1e-8 else None

    def recommend(self, user_id: int, k: int = 10) -> List[int]:
        if self._data is None or self.item_emb is None:
            raise RuntimeError("Model must be fit before recommending.")
        liked = self._data.user_pos_items.get(user_id, set())
        u_emb = self.encode_user(liked)
        if u_emb is None:
            return []
        scores = self.item_emb @ u_emb
        order = np.argsort(scores)[::-1]
        return [int(i) for i in order if i not in liked][:k]
