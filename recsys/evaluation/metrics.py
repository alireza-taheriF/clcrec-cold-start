"""Cold-start ranking metrics: HR@K and NDCG@K.

The evaluation protocol is preserved from the original ``src/evaluate.py``:

* the candidate pool is the set of *cold* items only,
* a user's ground truth is the cold items they liked,
* the user embedding is the mean of their *other* liked items' embeddings
  (ground-truth items are excluded to avoid leakage),
* items are ranked by cosine similarity (embeddings are already L2-normalized,
  so a dot product suffices).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def get_user_embedding(user_idx: int, item_emb: np.ndarray,
                       user_pos_items: dict,
                       exclude: Optional[set] = None,
                       user_encoder: Optional[object] = None) -> Optional[np.ndarray]:
    """User embedding via attention pooling (if provided) or mean pooling, L2-normalized.

    ``exclude`` removes ground-truth items so they cannot leak into the query
    vector. Returns ``None`` when the user has no usable history.
    """
    pos = user_pos_items.get(user_idx, set())
    train = pos - (exclude or set())
    if not train:
        return None
    train_idx = list(train)
    if user_encoder is not None:
        if hasattr(user_encoder, "encode_numpy"):
            return user_encoder.encode_numpy(item_emb[train_idx])
        if hasattr(user_encoder, "encode"):
            return user_encoder.encode(item_emb[train_idx])
    u_emb = item_emb[train_idx].mean(axis=0)
    norm = np.linalg.norm(u_emb)
    return u_emb / norm if norm > 1e-8 else None


def hit_rate_and_ndcg(user_idx: int, item_emb: np.ndarray,
                      user_pos_items: dict, eval_item_pool: np.ndarray,
                      K: int = 10,
                      user_encoder: Optional[object] = None):
    """Return ``(HR, NDCG)`` at ``K`` for one user, or ``(None, None)`` to skip.

    A user is skipped when they have no ground-truth cold items or no usable
    history to build a query vector from.
    """
    pos = user_pos_items.get(user_idx, set())
    gt = pos & set(eval_item_pool.tolist())
    if not gt:
        return None, None

    u_emb = get_user_embedding(user_idx, item_emb, user_pos_items, exclude=gt,
                               user_encoder=user_encoder)
    if u_emb is None:
        return None, None

    scores = item_emb[eval_item_pool] @ u_emb
    top_local = np.argsort(scores)[::-1][:K]
    top_items = set(eval_item_pool[top_local].tolist())

    hr = float(len(top_items & gt) > 0)

    dcg = sum(1.0 / np.log2(r + 2)
              for r, it in enumerate(eval_item_pool[top_local])
              if it in gt)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt), K)))
    ndcg = dcg / ideal if ideal > 0 else 0.0

    return hr, ndcg


def evaluate_embeddings(item_emb: np.ndarray, user_pos_items: dict,
                        test_users: list, cold_items: list,
                        ks: Optional[List[int]] = None,
                        user_encoder: Optional[object] = None,
                        verbose: bool = True) -> Dict[int, dict]:
    """Evaluate a single item-embedding table over the cold-item pool.

    Returns ``{K: {"HR": ..., "NDCG": ..., "n": ...}}``.
    """
    ks = ks or [5, 10, 20]
    pool = np.array(cold_items)
    results: Dict[int, dict] = {}

    for K in ks:
        hrs, ndcgs = [], []
        for u in test_users:
            hr, ndcg = hit_rate_and_ndcg(
                u, item_emb, user_pos_items, pool, K, user_encoder=user_encoder
            )
            if hr is not None:
                hrs.append(hr)
                ndcgs.append(ndcg)

        results[K] = {
            "HR": float(np.mean(hrs)) if hrs else 0.0,
            "NDCG": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "n": len(hrs),
        }

    if verbose:
        _print_table(results)
    return results


def _print_table(results: Dict[int, dict]) -> None:
    print(f"\n{'K':>4} | {'HR@K':>8} | {'NDCG@K':>8} | {'#Users':>7}")
    print("-" * 36)
    for K, res in results.items():
        print(f"{K:>4} | {res['HR']:>8.4f} | {res['NDCG']:>8.4f} | {res['n']:>7}")


def evaluate_all(embeddings_by_model: Dict[str, np.ndarray],
                 user_pos_items: dict, test_users: list, cold_items: list,
                 ks: Optional[List[int]] = None,
                 user_encoder: Optional[object] = None,
                 verbose: bool = True) -> Dict[str, Dict[int, dict]]:
    """Evaluate several models' embeddings with an identical protocol.

    ``embeddings_by_model`` maps a model name to its ``(n_movies, dim)`` item
    embedding table. Returns ``{model_name: {K: {...}}}``.
    """
    ks = ks or [5, 10, 20]
    out: Dict[str, Dict[int, dict]] = {}
    for name, emb in embeddings_by_model.items():
        if verbose:
            print(f"\n── {name} ──")
        cur_encoder = user_encoder if (user_encoder is not None and getattr(user_encoder, "emb_dim", emb.shape[1]) == emb.shape[1]) else None
        out[name] = evaluate_embeddings(
            emb, user_pos_items, test_users, cold_items, ks=ks,
            user_encoder=cur_encoder, verbose=verbose)
    return out
