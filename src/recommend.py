from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from src.dataset import ContentFeaturizer
from src.model import ContentEncoder


@dataclass
class ScoredItem:
    item_id: int
    score: float
    title: str = ""
    genres: str = ""


class ColdStartRecommender:
    """Inference API used by the lab CLI, HTTP service, and teaching demos."""

    def __init__(self, encoder: ContentEncoder, item_emb: np.ndarray,
                 featurizer: ContentFeaturizer, idx2movie: dict,
                 movie2idx: dict, meta: dict, warm_items, cold_items,
                 user_pos_items: dict | None = None,
                 user2idx: dict | None = None):
        self.encoder = encoder
        self.item_emb = item_emb
        self.featurizer = featurizer
        self.idx2movie = {int(k): int(v) for k, v in idx2movie.items()}
        self.movie2idx = {int(k): int(v) for k, v in movie2idx.items()}
        self.meta = meta
        self.warm = set(int(i) for i in warm_items)
        self.cold = set(int(i) for i in cold_items)
        self.user_pos_items = user_pos_items or {}
        self.user2idx = {int(k): int(v) for k, v in (user2idx or {}).items()}

    def _meta(self, movie_id: int) -> tuple[str, str]:
        info = self.meta.get(str(movie_id)) or self.meta.get(movie_id) or {}
        return info.get("title", ""), info.get("genres", "")

    def _pool(self, name: str) -> np.ndarray:
        if name == "cold":
            return np.array(sorted(self.cold), dtype=int)
        if name == "warm":
            return np.array(sorted(self.warm), dtype=int)
        return np.arange(len(self.item_emb), dtype=int)

    def _score_pool(self, query, pool, k, exclude=None):
        exclude = set(exclude or [])
        keep = np.array([i for i in pool if i not in exclude], dtype=int)
        if len(keep) == 0:
            return []
        scores = self.item_emb[keep] @ query
        top = np.argsort(-scores)[:k]
        out = []
        for local in top:
            idx = int(keep[local])
            movie_id = self.idx2movie[idx]
            title, genres = self._meta(movie_id)
            out.append(ScoredItem(movie_id, float(scores[local]), title, genres))
        return out

    def similar_items(self, movie_id: int, k: int = 10, pool: str = "all"):
        if movie_id not in self.movie2idx:
            raise KeyError(f"Unknown movie_id={movie_id}")
        idx = self.movie2idx[movie_id]
        query = self.item_emb[idx]
        return self._score_pool(query, self._pool(pool), k, exclude={idx})

    def recommend_from_likes(self, liked_movie_ids, k: int = 10, pool: str = "cold"):
        idxs = [self.movie2idx[i] for i in liked_movie_ids if i in self.movie2idx]
        if not idxs:
            raise ValueError("None of the liked items are in the catalog.")
        query = self.item_emb[idxs].mean(axis=0)
        norm = np.linalg.norm(query)
        if norm < 1e-8:
            raise ValueError("Degenerate user embedding.")
        query = query / norm
        return self._score_pool(query, self._pool(pool), k, exclude=set(idxs))

    def recommend_user(self, user_id: int, k: int = 10, pool: str = "cold"):
        if user_id not in self.user2idx:
            raise KeyError(f"Unknown user_id={user_id}")
        uidx = self.user2idx[user_id]
        liked_idx = self.user_pos_items.get(uidx, set())
        liked_ids = [self.idx2movie[i] for i in liked_idx]
        return self.recommend_from_likes(liked_ids, k=k, pool=pool)

    def embed_new_item(self, title: str, genres: str, year: float | None = None):
        x = self.featurizer.transform_one(title, genres, year)[None, :]
        z, _ = self.encoder.forward(x)
        return z[0]

    def recommend_for_new_item(self, title: str, genres: str,
                               year: float | None = None, k: int = 10):
        query = self.embed_new_item(title, genres, year)
        return self._score_pool(query, self._pool("all"), k)

    def save(self, artifact_dir: str) -> None:
        os.makedirs(artifact_dir, exist_ok=True)
        self.encoder.save(os.path.join(artifact_dir, "encoder.npz"))
        np.save(os.path.join(artifact_dir, "item_emb.npy"), self.item_emb)
        self.featurizer.save(os.path.join(artifact_dir, "featurizer.pkl"))
        payload = {
            "idx2movie": {str(k): v for k, v in self.idx2movie.items()},
            "movie2idx": {str(k): v for k, v in self.movie2idx.items()},
            "meta": {str(k): v for k, v in self.meta.items()},
            "warm": sorted(self.warm),
            "cold": sorted(self.cold),
            "user2idx": {str(k): v for k, v in self.user2idx.items()},
            "user_pos_items": {str(u): sorted(items)
                               for u, items in self.user_pos_items.items()},
        }
        with open(os.path.join(artifact_dir, "catalog.json"), "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, artifact_dir: str) -> "ColdStartRecommender":
        encoder = ContentEncoder.load(os.path.join(artifact_dir, "encoder.npz"))
        item_emb = np.load(os.path.join(artifact_dir, "item_emb.npy"))
        featurizer = ContentFeaturizer.load(os.path.join(artifact_dir, "featurizer.pkl"))
        with open(os.path.join(artifact_dir, "catalog.json")) as f:
            payload = json.load(f)
        user_pos = {int(u): set(v) for u, v in payload.get("user_pos_items", {}).items()}
        return cls(
            encoder=encoder,
            item_emb=item_emb,
            featurizer=featurizer,
            idx2movie={int(k): int(v) for k, v in payload["idx2movie"].items()},
            movie2idx={int(k): int(v) for k, v in payload["movie2idx"].items()},
            meta=payload["meta"],
            warm_items=payload["warm"],
            cold_items=payload["cold"],
            user_pos_items=user_pos,
            user2idx={int(k): int(v) for k, v in payload.get("user2idx", {}).items()},
        )
