import argparse

from src.config import ExperimentConfig
from src.experiment import run_experiment
from src.product import run_firstslot
from src.recommend import ColdStartRecommender
from src.serve import serve


def build_parser():
    p = argparse.ArgumentParser(
        description="FirstSlot — first-impression engine for zero-sale items",
    )
    p.add_argument("--product", action="store_true",
                   help="Train FirstSlot on the industrial MRO catalog (what you show a buyer)")
    p.add_argument("--protocol", choices=("academic", "legacy"), default="academic",
                   help="academic: no leakage, warm-only user history, Recall+CI; "
                        "legacy: original README cold-pool HR table")
    p.add_argument("--epochs", type=int, default=None,
                   help="Default 40 for --product, 150 for MovieLens")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-test-users", dest="n_test_users", type=int, default=1000)
    p.add_argument("--cold-threshold", dest="cold_threshold", type=int, default=10)
    p.add_argument("--cold-ratio", dest="cold_ratio", type=float, default=None)
    p.add_argument("--no-title", dest="use_title", action="store_false")
    p.add_argument("--artifact-dir", dest="artifact_dir", default="artifacts")
    p.add_argument("--data-dir", dest="data_dir", default="data")
    p.add_argument("--serve", action="store_true",
                   help="Load artifacts and start the HTTP API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--demo-new-item", action="store_true",
                   help="Encode a brand-new item and print nearest catalog neighbors")
    p.add_argument("--title", default="Dune (2021)")
    p.add_argument("--genres", default="Action|Adventure|Sci-Fi")
    p.add_argument("--year", type=float, default=None)
    p.add_argument("--k", type=int, default=8)
    return p


def demo_new_item(artifact_dir, title, genres, year, k):
    rec = ColdStartRecommender.load(artifact_dir)
    print(f"New item: {title}  |  {genres}")
    for row in rec.recommend_for_new_item(title, genres, year, k=k):
        tag = "COLD" if rec.movie2idx[row.item_id] in rec.cold else "warm"
        print(f"  {row.score:7.4f}  [{tag}]  {row.title}  ({row.genres})")


def main():
    args = build_parser().parse_args()
    if args.product:
        run_firstslot(args.artifact_dir, epochs=args.epochs or 40, seed=args.seed)
        return
    if args.serve:
        serve(args.artifact_dir, args.host, args.port)
        return
    if args.demo_new_item:
        demo_new_item(args.artifact_dir, args.title, args.genres, args.year, args.k)
        return

    cfg = ExperimentConfig(
        data_dir=args.data_dir,
        artifact_dir=args.artifact_dir,
        epochs=args.epochs or 150,
        seed=args.seed,
        n_test_users=args.n_test_users,
        cold_threshold=args.cold_threshold,
        cold_ratio=args.cold_ratio,
        use_title=args.use_title,
        protocol=args.protocol,
    )
    run_experiment(cfg)


if __name__ == "__main__":
    main()
