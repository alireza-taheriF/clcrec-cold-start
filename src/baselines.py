from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return (x / norms).astype(np.float32)


def raw_content_embeddings(content_features: np.ndarray) -> np.ndarray:
    return l2_normalize(content_features)


def ridge_content_map(content_features: np.ndarray,
                      collab_emb: np.ndarray,
                      warm_items,
                      l2: float = 1.0) -> np.ndarray:
    """Linear constraint-style baseline: content -> collaborative space (ridge)."""
    warm = np.asarray(warm_items)
    X = content_features[warm]
    Y = collab_emb[warm]
    xtx = X.T @ X
    xtx.flat[:: xtx.shape[0] + 1] += l2
    W = np.linalg.solve(xtx, X.T @ Y)
    mapped = content_features @ W
    return l2_normalize(mapped)
