"""Attention-based user profiling for Cold-Start Recommenders.

Replaces uniform mean-pooling with learned or feature-aware attention pooling,
weighting the most salient items in a user's interaction history.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class CentralityAttentionUserEncoder:
    """Attention pooling weighting items by their centrality / alignment with user preferences.

    Items closer to the user's core interest cluster receive higher attention weight,
    filtering out noise and accidental interactions.
    """

    def __init__(self, emb_dim: int = 32, tau: float = 0.5):
        self.emb_dim = emb_dim
        self.tau = tau

    def encode(self, item_vecs: np.ndarray) -> Optional[np.ndarray]:
        if len(item_vecs) == 0:
            return None
        if len(item_vecs) == 1:
            norm = np.linalg.norm(item_vecs[0])
            return item_vecs[0] / (norm + 1e-8)

        # Center vector (prototype)
        center = item_vecs.mean(axis=0)
        c_norm = np.linalg.norm(center)
        if c_norm > 1e-8:
            center = center / c_norm

        # Scaled dot-product attention against prototype
        scores = (item_vecs @ center) / self.tau
        exp_s = np.exp(scores - np.max(scores))
        weights = exp_s / (np.sum(exp_s) + 1e-8)

        user_emb = np.sum(item_vecs * weights[:, None], axis=0)
        norm = np.linalg.norm(user_emb)
        return user_emb / (norm + 1e-8)

    def encode_numpy(self, item_vecs: np.ndarray) -> Optional[np.ndarray]:
        return self.encode(item_vecs)


if HAS_TORCH:
    class TorchAttentionUserEncoder(nn.Module):
        """PyTorch attention pooling layer with centrality-guided scoring."""

        def __init__(self, emb_dim: int = 32, tau: float = 0.5):
            super().__init__()
            self.emb_dim = emb_dim
            self.tau = tau

        def forward(self, item_vecs: torch.Tensor) -> torch.Tensor:
            if item_vecs.shape[0] == 0:
                raise ValueError("Cannot pool empty item set.")
            if item_vecs.shape[0] == 1:
                return F.normalize(item_vecs[0], p=2, dim=0)

            center = F.normalize(item_vecs.mean(dim=0, keepdim=True), p=2, dim=1) # (1, d)
            scores = (item_vecs @ center.t()).squeeze(-1) / self.tau # (n,)
            weights = F.softmax(scores, dim=0).unsqueeze(-1) # (n, 1)

            user_emb = torch.sum(weights * item_vecs, dim=0)
            return F.normalize(user_emb, p=2, dim=0)

        def encode_numpy(self, item_vecs: np.ndarray) -> np.ndarray:
            self.eval()
            with torch.no_grad():
                t = torch.from_numpy(item_vecs.astype(np.float32))
                return self.forward(t).cpu().numpy()
