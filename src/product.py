"""Train FirstSlot on the MRO catalog and write merchant artifacts."""

from __future__ import annotations

import json
import os

from src.config import ExperimentConfig
from src.dataset import (
    align_content_features,
    build_collab_embeddings,
    build_content_features,
    build_interactions,
)
from src.launch import policy_compare
from src.mro import generate_mro, to_movies_frame, to_ratings_frame
from src.recommend import ColdStartRecommender
from src.train import train


def run_firstslot(artifact_dir: str = "artifacts", epochs: int = 40, seed: int = 7) -> dict:
    os.makedirs(artifact_dir, exist_ok=True)
    world = generate_mro(seed=seed)

    movies = to_movies_frame(world.items)
    ratings = to_ratings_frame(world.events)
    content_matrix, movie_id_list, feature_names, featurizer = build_content_features(
        movies, use_title=True, title_svd_dim=12,
    )
    user_pos_items, user2idx, movie2idx, idx2movie = build_interactions(
        ratings, pos_threshold=1,
        all_item_ids=world.items["item_id"].tolist(),
        all_user_ids=world.users["user_id"].tolist(),
    )

    n_users = len(user2idx)
    n_items = len(movie2idx)
    content_features = align_content_features(
        content_matrix, movie_id_list, movie2idx, len(feature_names),
    )

    cold_ids = set(world.items.loc[world.items["is_cold"] == 1, "item_id"].tolist())
    warm_items = [i for mid, i in movie2idx.items() if mid not in cold_ids]
    cold_items = [i for mid, i in movie2idx.items() if mid in cold_ids]

    print(f"FirstSlot MRO | plants={n_users} | parts={n_items} | "
          f"warm={len(warm_items)} | new={len(cold_items)} | events={len(world.events)}")

    collab = build_collab_embeddings(
        user_pos_items, n_users, n_items, emb_dim=24, exclude_items=cold_items,
    )
    cfg = ExperimentConfig(epochs=epochs, hidden_dim=32, emb_dim=24, batch_size=64,
                           lr=0.005, tau=0.15, l2_reg=5e-4, patience=12, seed=seed)
    encoder, history = train(
        content_features, collab, warm_items,
        emb_dim=cfg.emb_dim, hidden_dim=cfg.hidden_dim, epochs=cfg.epochs,
        batch_size=cfg.batch_size, lr=cfg.lr, tau=cfg.tau, l2_reg=cfg.l2_reg,
        seed=cfg.seed, patience=cfg.patience, val_warm_frac=0.12,
    )
    item_emb = encoder.encode_all(content_features)

    item_meta = {}
    item_category = {}
    for row in world.items.itertuples(index=False):
        item_meta[int(row.item_id)] = {
            "title": row.title,
            "genres": row.tags,
            "tags": row.tags,
            "sku": row.sku,
            "category": row.category,
            "category_fa": row.category_fa,
            "brand": row.brand,
            "size": row.size,
            "family": row.family,
            "is_cold": int(row.is_cold),
        }
        if row.item_id in movie2idx:
            item_category[movie2idx[int(row.item_id)]] = row.category

    user_meta = {
        int(row.user_id): {"name": row.name, "segment": row.segment, "segment_fa": row.segment_fa}
        for row in world.users.itertuples(index=False)
    }
    idx_segments = {}
    for uid, info in user_meta.items():
        if uid in user2idx:
            idx_segments[user2idx[uid]] = info["segment"]

    truth_by_item = {}
    for row in world.truth.itertuples(index=False):
        if row.item_id not in movie2idx or row.user_id not in user2idx:
            continue
        truth_by_item.setdefault(movie2idx[int(row.item_id)], set()).add(user2idx[int(row.user_id)])

    compare_rows, eligible = policy_compare(
        item_emb, user_pos_items, warm_items, cold_items, truth_by_item,
        item_category, idx_segments, budgets=(20, 40, 80), diversity=0.0, seed=seed,
    )

    print("\nLaunch efficiency (mean over new SKUs) — precision of first B impressions")
    print(f"{'B':>4} | {'random':>8} | {'popular':>8} | {'category':>8} | {'FirstSlot':>9}")
    print("-" * 56)
    by_b = {}
    for row in compare_rows:
        by_b.setdefault(row["budget"], {})[row["policy"]] = row
    for B in (20, 40, 80):
        cell = by_b[B]
        print(f"{B:>4} | {cell['random']['precision']:.3f}   | {cell['popular']['precision']:.3f}   | "
              f"{cell['category']['precision']:.3f}   | {cell['firstslot']['precision']:.3f}")

    rec = ColdStartRecommender(
        encoder=encoder,
        item_emb=item_emb,
        featurizer=featurizer,
        idx2movie=idx2movie,
        movie2idx=movie2idx,
        meta=item_meta,
        warm_items=warm_items,
        cold_items=cold_items,
        user_pos_items=user_pos_items,
        user2idx=user2idx,
        user_meta=user_meta,
        vertical="mro",
        truth_by_item={str(k): sorted(v) for k, v in truth_by_item.items()},
    )
    rec.save(artifact_dir)

    report = {
        "product": "FirstSlot",
        "vertical": "mro",
        "n_plants": n_users,
        "n_parts": n_items,
        "n_events": int(len(world.events)),
        "n_new": len(cold_items),
        "n_eval_skus": len(eligible),
        "history": [float(x) for x in history],
        "launch": compare_rows,
        "leakage_guard": True,
    }
    with open(os.path.join(artifact_dir, "launch_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFirstSlot artifacts → {artifact_dir}/")
    return report
