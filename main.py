import numpy as np
import matplotlib.pyplot as plt

from src.dataset  import (download_movielens, load_movielens,
                           build_content_features, build_interactions,
                           build_collab_embeddings, split_cold_warm)
from src.train    import train
from src.evaluate import evaluate


DATA_DIR       = "data"
EMB_DIM        = 32
HIDDEN_DIM     = 32
COLD_THRESHOLD = 10
EPOCHS         = 150
BATCH_SIZE     = 256
LR             = 0.005
TAU            = 0.1
L2_REG         = 5e-4
N_TEST_USERS   = 1000
SEED           = 42


def main():
    download_movielens(DATA_DIR)
    ratings, movies = load_movielens(DATA_DIR)

    content_matrix, movie_id_list, genres = build_content_features(movies)
    user_pos_items, user2idx, movie2idx, idx2movie = build_interactions(ratings)

    n_users  = len(user2idx)
    n_movies = len(movie2idx)

    content_features = np.zeros((n_movies, len(genres)), dtype="float32")
    for raw_idx, movie_id in enumerate(movie_id_list):
        if movie_id in movie2idx:
            content_features[movie2idx[movie_id]] = content_matrix[raw_idx]

    print(f"Users: {n_users} | Movies: {n_movies} | Genres: {len(genres)}")

    print("Building collaborative embeddings...")
    item_collab_emb = build_collab_embeddings(user_pos_items, n_users,
                                               n_movies, EMB_DIM)

    warm_items, cold_items = split_cold_warm(user_pos_items, n_movies,
                                              COLD_THRESHOLD)
    print(f"Warm: {len(warm_items)} | Cold: {len(cold_items)}")

    encoder, history = train(
        content_features = content_features,
        item_collab_emb  = item_collab_emb,
        warm_items       = warm_items,
        emb_dim          = EMB_DIM,
        hidden_dim       = HIDDEN_DIM,
        epochs           = EPOCHS,
        batch_size       = BATCH_SIZE,
        lr               = LR,
        tau              = TAU,
        l2_reg           = L2_REG,
        seed             = SEED,
    )

    all_content_emb = encoder.encode_all(content_features)

    np.random.seed(SEED)
    test_users = list(np.random.choice(list(user_pos_items.keys()),
                                        size=N_TEST_USERS, replace=False))

    print("\n── CLCRec (Contrastive) ──")
    results_cl = evaluate(all_content_emb, user_pos_items,
                           test_users, cold_items)

    content_norm = content_features / (
        np.linalg.norm(content_features, axis=1, keepdims=True) + 1e-8)
    print("\n── Baseline (Raw Genre Vectors) ──")
    results_base = evaluate(content_norm, user_pos_items,
                             test_users, cold_items)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    Ks = [5, 10, 20]

    ax1.plot(Ks, [results_cl[k]["HR"]   for k in Ks], "o-", label="CLCRec")
    ax1.plot(Ks, [results_base[k]["HR"] for k in Ks], "s--", label="Baseline")
    ax1.set_title("Hit Rate@K (Cold Items)"); ax1.set_xlabel("K")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(range(1, len(history)+1), history, color="teal")
    ax2.set_title("Training Loss"); ax2.set_xlabel("Epoch")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\nChart saved to results.png")


if __name__ == "__main__":
    main()
