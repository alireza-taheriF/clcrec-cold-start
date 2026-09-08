from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from src.baselines import raw_content_embeddings, ridge_content_map
from src.config import ExperimentConfig
from src.dataset import (
    align_content_features,
    build_collab_embeddings,
    build_content_features,
    build_interactions,
    download_movielens,
    load_movielens,
    split_cold_warm,
    split_cold_warm_random,
)
from src.evaluate import build_user_splits, evaluate, evaluate_academic
from src.recommend import ColdStartRecommender
from src.train import train


def _select_test_users(user_pos_items, n_test_users, seed):
    rng = np.random.default_rng(seed)
    keys = list(user_pos_items.keys())
    n = min(n_test_users, len(keys))
    return list(rng.choice(keys, size=n, replace=False))


def _plot(path, history, academic_tables=None, legacy_tables=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    Ks = [5, 10, 20]

    ax1 = axes[0]
    if academic_tables:
        for name, table in academic_tables.items():
            ax1.plot(Ks, [table[k]["Recall"] for k in Ks], marker="o", label=name)
        ax1.set_title("Recall@K — cold items (academic)")
    elif legacy_tables:
        for name, table in legacy_tables.items():
            ax1.plot(Ks, [table[k]["HR"] for k in Ks], marker="o", label=name)
        ax1.set_title("Hit Rate@K (legacy protocol)")
    ax1.set_xlabel("K")
    ax1.legend()
    ax1.grid(alpha=0.3)

    axes[1].plot(range(1, len(history) + 1), history, color="teal")
    axes[1].set_title("Training Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_experiment(cfg: ExperimentConfig) -> dict:
    os.makedirs(cfg.artifact_dir, exist_ok=True)

    download_movielens(cfg.data_dir)
    ratings, movies = load_movielens(cfg.data_dir)

    content_matrix, movie_id_list, feature_names, featurizer = build_content_features(
        movies, use_title=cfg.use_title, title_svd_dim=cfg.title_svd_dim,
    )
    user_pos_items, user2idx, movie2idx, idx2movie = build_interactions(ratings)

    n_users = len(user2idx)
    n_movies = len(movie2idx)
    content_features = align_content_features(
        content_matrix, movie_id_list, movie2idx, len(feature_names),
    )

    if cfg.cold_ratio:
        warm_items, cold_items = split_cold_warm_random(
            n_movies, cfg.cold_ratio, cfg.seed,
        )
        split_name = f"random ratio={cfg.cold_ratio}"
    else:
        warm_items, cold_items = split_cold_warm(
            user_pos_items, n_movies, cfg.cold_threshold,
        )
        split_name = f"threshold<{cfg.cold_threshold}"

    print(f"Users: {n_users} | Movies: {n_movies} | Features: {len(feature_names)}")
    print(f"Warm: {len(warm_items)} | Cold: {len(cold_items)} | split={split_name}")
    print(f"Protocol: {cfg.protocol} | title_features={cfg.use_title}")

    print("Building collaborative embeddings on warm items only (no leakage)...")
    item_collab_emb = build_collab_embeddings(
        user_pos_items, n_users, n_movies, cfg.emb_dim, exclude_items=cold_items,
    )

    encoder, history = train(
        content_features=content_features,
        item_collab_emb=item_collab_emb,
        warm_items=warm_items,
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        tau=cfg.tau,
        l2_reg=cfg.l2_reg,
        seed=cfg.seed,
        patience=cfg.patience,
        val_warm_frac=cfg.val_warm_frac,
    )

    clc_emb = encoder.encode_all(content_features)
    raw_emb = raw_content_embeddings(content_features)
    ridge_emb = ridge_content_map(content_features, item_collab_emb, warm_items)

    test_users = _select_test_users(user_pos_items, cfg.n_test_users, cfg.seed)
    history_u, truth_cold, truth_warm = build_user_splits(
        user_pos_items, warm_items, cold_items,
    )

    report = {
        "config": cfg.to_dict(),
        "n_users": n_users,
        "n_movies": n_movies,
        "n_features": len(feature_names),
        "n_warm": len(warm_items),
        "n_cold": len(cold_items),
        "split": split_name,
        "leakage_guard": True,
        "models": {},
    }

    models = {
        "CLCRec": clc_emb,
        "Ridge-Map": ridge_emb,
        "Raw-Content": raw_emb,
    }

    if cfg.protocol == "legacy":
        legacy_tables = {}
        for name, emb in models.items():
            print(f"\n── {name} (legacy cold-pool) ──")
            legacy_tables[name] = evaluate(emb, user_pos_items, test_users, cold_items, cfg.ks)
        report["legacy"] = {k: {str(kk): vv for kk, vv in table.items()}
                            for k, table in legacy_tables.items()}
        chart_tables = None
        _plot(os.path.join(cfg.artifact_dir, "results.png"), history,
              legacy_tables=legacy_tables)
        _plot("results.png", history, legacy_tables=legacy_tables)
    else:
        academic = {}
        for name, emb in models.items():
            academic[name] = {
                "cold": evaluate_academic(
                    emb, history_u, truth_cold, test_users, cold_items,
                    cfg.ks, cfg.n_bootstrap, cfg.seed,
                    title=f"{name} · COLD (history=warm only)",
                ),
                "all": evaluate_academic(
                    emb, history_u, truth_cold, test_users,
                    list(range(n_movies)),
                    cfg.ks, cfg.n_bootstrap, cfg.seed,
                    title=f"{name} · ALL-item ranking of cold GT",
                ),
            }
        report["academic"] = {
            name: {scen: {str(k): v for k, v in table.items()}
                   for scen, table in scenarios.items()}
            for name, scenarios in academic.items()
        }
        cold_recall = {name: academic[name]["cold"] for name in models}
        _plot(os.path.join(cfg.artifact_dir, "results.png"), history,
              academic_tables=cold_recall)
        _plot("results.png", history, academic_tables=cold_recall)

    meta = {
        int(row.movie_id): {"title": row.title, "genres": row.genres}
        for row in movies.itertuples(index=False)
    }

    recommender = ColdStartRecommender(
        encoder=encoder,
        item_emb=clc_emb,
        featurizer=featurizer,
        idx2movie=idx2movie,
        movie2idx=movie2idx,
        meta=meta,
        warm_items=warm_items,
        cold_items=cold_items,
        user_pos_items=user_pos_items,
        user2idx=user2idx,
    )
    recommender.save(cfg.artifact_dir)

    with open(os.path.join(cfg.artifact_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nArtifacts written to {cfg.artifact_dir}/")
    print("Chart saved to results.png")
    return report
