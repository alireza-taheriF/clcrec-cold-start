"""Non-learned baselines for cold-start recommendation.

Both baselines produce an ``(n_movies, dim)`` item-embedding table that plugs
straight into :func:`recsys.evaluation.metrics.evaluate_embeddings`, so they are
scored with exactly the same protocol as the trained encoders.

* :func:`content_knn_embeddings` — L2-normalized raw content (genre + year).
  Cosine similarity in this space is a content-only KNN recommender.
* :func:`svd_cf_embeddings` — the SVD collaborative embeddings themselves. This
  is the pure collaborative-filtering reference; it is *expected to be weak on
  cold items*, because cold items have too few interactions for SVD to place
  them well — which is the whole motivation for the content encoder.
"""
from __future__ import annotations

import numpy as np

from recsys.data.utils import l2_normalize


def content_knn_embeddings(content_features: np.ndarray) -> np.ndarray:
    """Content-only baseline: L2-normalized raw genre+year vectors."""
    return l2_normalize(content_features, axis=1)


def svd_cf_embeddings(item_collab_emb: np.ndarray) -> np.ndarray:
    """Pure collaborative-filtering baseline: the SVD item embeddings as-is.

    They are already L2-normalized upstream; we re-normalize defensively so the
    dot-product scoring in the metrics stays a true cosine similarity.
    """
    return l2_normalize(item_collab_emb, axis=1)


def build_baseline_embeddings(data) -> dict:
    """Return ``{name: embedding_table}`` for every baseline, given a DataBundle."""
    return {
        "Content-KNN": content_knn_embeddings(data.content_features),
        "SVD-CF": svd_cf_embeddings(data.item_collab_emb),
    }
