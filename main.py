"""CLI entrypoint: train, evaluate, and save artifacts for either backend.

Usage
-----
    python main.py --backend numpy
    python main.py --backend torch --epochs 100 --learnable-tau
    python main.py --backend numpy --save-artifacts   # write files for the API

The heavy pipeline (data prep -> SVD -> train -> encode -> evaluate -> plot)
runs here. Both backends share the same data bundle, evaluation protocol, and
plotting so their numbers are directly comparable. The original from-scratch
code in ``src/`` is left untouched as the teaching reference; this CLI drives
the refactored package in ``recsys/``.
"""
from __future__ import annotations

import argparse

import numpy as np

from config import Config
from recsys.data.dataset import prepare_data
from recsys.data.artifacts import save_artifacts
from recsys.evaluation.baselines import build_baseline_embeddings
from recsys.evaluation.metrics import evaluate_all
from recsys.evaluation.logging_utils import (
    write_metrics_json,
    write_metrics_csv,
    write_metrics_summary,
)


CONTRASTIVE_NAME = "CLCRec (Contrastive)"


def build_model(config: Config):
    """Instantiate the recommender for the configured backend."""
    if getattr(config, "joint_training", False) or config.backend == "joint":
        from recsys.models.joint_model import JointCLCRecRecommender
        return JointCLCRecRecommender(config)
    if config.backend == "numpy":
        from recsys.models.numpy_encoder import NumpyRecommender
        return NumpyRecommender(config)
    if config.backend == "torch":
        from recsys.models.torch_encoder import TorchRecommender
        return TorchRecommender(config)
    raise ValueError(f"Unknown backend: {config.backend!r} (use 'numpy', 'torch', or 'joint').")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CLCRec cold-start recommender.")
    p.add_argument("--backend", choices=["numpy", "torch", "joint"], default="numpy",
                   help="Which encoder implementation to train.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--emb-dim", type=int, default=None)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--learnable-tau", action="store_true",
                   help="torch only: learn the InfoNCE temperature.")
    p.add_argument("--n-test-users", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None,
                   help="torch only: compute device (default auto).")
    p.add_argument("--save-artifacts", action="store_true",
                   help="Persist item embeddings + metadata for the API.")
    p.add_argument("--no-plot", action="store_true", help="Skip writing results.png.")
    p.add_argument("--attention", action="store_true",
                   help="Innovation 2: Enable attention-based user profiling.")
    p.add_argument("--hard-negatives", action="store_true",
                   help="Innovation 3: Enable content-aware hard negative mining.")
    p.add_argument("--joint-training", action="store_true",
                   help="Innovation 5: Enable end-to-end joint BPR + InfoNCE training.")
    p.add_argument("--no-text", action="store_true",
                   help="Disable text semantic features (revert to genre+year only).")
    return p.parse_args()


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Fold any provided CLI flags onto the default config."""
    config.backend = args.backend
    if args.epochs is not None:        config.epochs = args.epochs
    if args.batch_size is not None:    config.batch_size = args.batch_size
    if args.lr is not None:            config.lr = args.lr
    if args.emb_dim is not None:       config.emb_dim = args.emb_dim
    if args.tau is not None:           config.tau = args.tau
    if args.learnable_tau:             config.learnable_tau = True
    if args.n_test_users is not None:  config.n_test_users = args.n_test_users
    if args.device is not None:        config.device = args.device
    if args.attention:                 config.user_attention = True
    if args.hard_negatives:            config.hard_negatives = True
    if args.joint_training:            config.joint_training = True
    if args.no_text:                   config.use_text_features = False
    return config


def sample_test_users(user_pos_items: dict, n: int, seed: int) -> list:
    """Reproducibly sample up to ``n`` users to evaluate on."""
    np.random.seed(seed)
    users = list(user_pos_items.keys())
    n = min(n, len(users))
    return list(np.random.choice(users, size=n, replace=False))


def plot_results(results: dict, history: list, ks: list, out_path: str) -> None:
    """Two-panel figure: HR@K across models + the contrastive training curve."""
    import matplotlib
    matplotlib.use("Agg")           # headless-safe backend
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    markers = {"CLCRec (Contrastive)": "o-", "Content-KNN": "s--", "SVD-CF": "^:"}
    for name, per_k in results.items():
        hr = [per_k[k]["HR"] for k in ks]
        ax1.plot(ks, hr, markers.get(name, "d-"), label=name)
    ax1.set_title("Hit Rate@K (Cold Items)")
    ax1.set_xlabel("K"); ax1.set_ylabel("HR@K")
    ax1.legend(); ax1.grid(alpha=0.3)

    if history:
        ax2.plot(range(1, len(history) + 1), history, color="teal")
    ax2.set_title("Contrastive Training Loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("InfoNCE loss")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nChart saved to {out_path}")


def main() -> None:
    args = parse_args()
    config = apply_overrides(Config(), args)
    config.paths.ensure()

    # 1. Data pipeline (download is skipped when data already exists locally).
    data = prepare_data(config)
    print(f"Users: {data.n_users} | Movies: {data.n_movies} | "
          f"Features: {data.content_dim}")
    print(f"Warm: {len(data.warm_items)} | Cold: {len(data.cold_items)}")

    # 2. Train the contrastive encoder for the chosen backend.
    model = build_model(config)
    model.fit(data)

    # 3. Evaluate the contrastive model and both baselines with one protocol.
    test_users = sample_test_users(data.user_pos_items, config.n_test_users,
                                   config.seed)
    embeddings = {CONTRASTIVE_NAME: model.item_emb}
    embeddings.update(build_baseline_embeddings(data))

    user_encoder = None
    if config.user_attention:
        from recsys.models.attention import TorchAttentionUserEncoder, CentralityAttentionUserEncoder
        if config.backend == "torch":
            user_encoder = TorchAttentionUserEncoder(emb_dim=config.emb_dim)
        else:
            user_encoder = CentralityAttentionUserEncoder(emb_dim=config.emb_dim)
        print("Using Attention-based User Profiling.")

    results = evaluate_all(embeddings, data.user_pos_items, test_users,
                           data.cold_items, ks=config.ks, user_encoder=user_encoder)

    # 4. Persist metrics (JSON + CSV + regenerated Markdown summary).
    payload = {
        "config": config.to_dict(),
        "n_test_users_effective": len(test_users),
        "results": {name: {str(k): v for k, v in per_k.items()}
                    for name, per_k in results.items()},
    }
    write_metrics_json(config.paths.metrics_json, payload)
    write_metrics_csv(config.paths.results_dir + "/metrics.csv",
                      payload["results"], config.ks)
    write_metrics_summary(config.paths.metrics_summary, payload["results"],
                          config.ks, config.to_dict())
    print(f"\nMetrics written to {config.paths.metrics_json}")

    # 5. Optional: plot and save artifacts for the API.
    if not args.no_plot:
        plot_results(results, model.history, config.ks, config.paths.results_png)

    if args.save_artifacts:
        save_artifacts(
            config.paths.item_embeddings, config.paths.metadata,
            item_emb=model.item_emb,
            user_pos_items=data.user_pos_items,
            titles=data.titles, genres=data.genres,
            idx2movie=data.idx2movie, backend=config.backend,
        )
        print(f"Artifacts saved to {config.paths.artifacts_dir}")


if __name__ == "__main__":
    main()
