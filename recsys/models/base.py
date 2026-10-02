"""Shared model interface.

Every recommender in this project — the from-scratch NumPy encoder, the PyTorch
encoder, and the evaluation baselines — is usable through the same small surface
so that the trainer, evaluator, and API don't care which backend they hold.
"""
from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RecommenderModel(Protocol):
    """Minimal interface shared by all recommenders.

    Implementations map content features into an embedding space aligned with
    collaborative signals, then score items by cosine similarity to a user
    vector built from that user's liked items.
    """

    def fit(self, data, val_data=None) -> "RecommenderModel":
        """Train on a :class:`~recsys.data.dataset.DataBundle`."""
        ...

    def encode_items(self, item_content_matrix: np.ndarray) -> np.ndarray:
        """Return L2-normalized embeddings for every row of the content matrix."""
        ...

    def encode_user(self, liked_item_indices) -> Optional[np.ndarray]:
        """Build a single user embedding from the indices of that user's liked items."""
        ...

    def recommend(self, user_id: int, k: int) -> List[int]:
        """Return the top-``k`` recommended internal item indices for a user."""
        ...
