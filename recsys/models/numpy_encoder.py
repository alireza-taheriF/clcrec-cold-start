"""From-scratch NumPy content encoder with hand-written backprop.

This is the *teaching core* of the project: every gradient here is derived and
written by hand, with no autograd. The PyTorch version in
:mod:`recsys.models.torch_encoder` mirrors this same objective so the two can be
compared directly. Keep this file readable — clarity matters more than cleverness.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class ContentEncoder:
    """Two-layer MLP: ``content_dim -> hidden_dim -> emb_dim`` with L2-normalized output.

    Weights are trained with a hand-written Adam optimizer against the gradient
    supplied by :func:`infonce_loss`.
    """

    def __init__(self, content_dim: int, emb_dim: int,
                 hidden_dim: int = 64, lr: float = 0.005):

        scale1  = np.sqrt(2.0 / (content_dim + hidden_dim))
        self.W1 = (np.random.randn(content_dim, hidden_dim) * scale1).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        scale2  = np.sqrt(2.0 / (hidden_dim + emb_dim))
        self.W2 = (np.random.randn(hidden_dim, emb_dim) * scale2).astype(np.float32)
        self.b2 = np.zeros(emb_dim, dtype=np.float32)

        self.lr = lr

        # Adam moment buffers, one pair per parameter tensor.
        self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1); self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2); self.vb2 = np.zeros_like(self.b2)
        self.t   = 0

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_grad(self, x):
        return (x > 0).astype(np.float32)

    def forward(self, x: np.ndarray):
        h_pre  = x  @ self.W1 + self.b1
        h      = self._relu(h_pre)

        z_pre  = h  @ self.W2 + self.b2

        norms  = np.linalg.norm(z_pre, axis=1, keepdims=True) + 1e-8
        z_norm = z_pre / norms

        cache = (x, h_pre, h, z_pre, norms)
        return z_norm, cache

    def backward(self, grad_z_norm: np.ndarray, cache,
                 l2_reg: float = 1e-4):
        x, h_pre, h, z_pre, norms = cache

        # Backprop through the L2-normalization: d(z/||z||).
        dot      = (grad_z_norm * z_pre).sum(axis=1, keepdims=True)
        grad_pre = (grad_z_norm - z_pre * dot / norms**2) / norms

        dW2 = h.T @ grad_pre + l2_reg * self.W2
        db2 = grad_pre.sum(axis=0)
        dh  = grad_pre @ self.W2.T

        dh_pre = dh * self._relu_grad(h_pre)

        dW1 = x.T @ dh_pre + l2_reg * self.W1
        db1 = dh_pre.sum(axis=0)

        return dW1, db1, dW2, db2

    def _adam(self, param, m, v, grad, t, lr):
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad**2
        m_h = m / (1 - b1**t)
        v_h = v / (1 - b2**t)
        param -= lr * m_h / (np.sqrt(v_h) + eps)
        return param, m, v

    def adam_step(self, dW1, db1, dW2, db2):
        self.t += 1
        self.W1, self.mW1, self.vW1 = self._adam(self.W1, self.mW1, self.vW1, dW1, self.t, self.lr)
        self.b1, self.mb1, self.vb1 = self._adam(self.b1, self.mb1, self.vb1, db1, self.t, self.lr)
        self.W2, self.mW2, self.vW2 = self._adam(self.W2, self.mW2, self.vW2, dW2, self.t, self.lr)
        self.b2, self.mb2, self.vb2 = self._adam(self.b2, self.mb2, self.vb2, db2, self.t, self.lr)

    def encode_all(self, features: np.ndarray, batch_size: int = 512):
        results = []
        for start in range(0, len(features), batch_size):
            z, _ = self.forward(features[start: start + batch_size])
            results.append(z)
        return np.vstack(results)


def infonce_loss(z_content: np.ndarray, z_collab: np.ndarray,
                 tau: float = 0.1):
    """InfoNCE loss aligning content embeddings to fixed collaborative embeddings.

    For a batch of warm items the positive pair is ``(z_content[i], z_collab[i])``;
    every other collaborative embedding in the batch is a negative. Returns the
    scalar loss and the gradient w.r.t. ``z_content``.
    """
    B = z_content.shape[0]

    logits = z_content @ z_collab.T / tau

    logits -= logits.max(axis=1, keepdims=True)

    exp_l   = np.exp(logits)
    sum_exp = exp_l.sum(axis=1, keepdims=True)

    log_prob = logits - np.log(sum_exp)

    loss = -np.mean(np.diag(log_prob))

    softmax_p = exp_l / sum_exp
    softmax_p[np.arange(B), np.arange(B)] -= 1.0
    grad = (softmax_p @ z_collab) / (tau * B)

    return loss, grad


class NumpyRecommender:
    """:class:`~recsys.models.base.RecommenderModel` adapter over :class:`ContentEncoder`.

    Wraps the raw encoder with the shared fit/encode/recommend interface and keeps
    the item embedding table + user-history lookup needed to serve recommendations.
    """

    backend = "numpy"

    def __init__(self, config):
        self.config = config
        self.encoder: Optional[ContentEncoder] = None
        self.item_emb: Optional[np.ndarray] = None   # (n_movies, emb_dim), populated after fit
        self.history: List[float] = []
        self._data = None

    def fit(self, data, val_data=None) -> "NumpyRecommender":
        from recsys.training.trainer_numpy import train_numpy
        self.encoder, self.history = train_numpy(data, self.config)
        self.item_emb = self.encode_items(data.content_features)
        self._data = data
        return self

    def encode_items(self, item_content_matrix: np.ndarray) -> np.ndarray:
        if self.encoder is None:
            raise RuntimeError("Model must be fit before encoding items.")
        return self.encoder.encode_all(item_content_matrix)

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
        # Filter out items the user already liked, then take the top-k.
        return [int(i) for i in order if i not in liked][:k]
