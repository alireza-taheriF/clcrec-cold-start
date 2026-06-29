import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


def download_movielens(data_dir: str = "data"):
    os.makedirs(data_dir, exist_ok=True)
    url      = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = os.path.join(data_dir, "ml-1m.zip")

    if not os.path.exists(os.path.join(data_dir, "ml-1m", "ratings.dat")):
        print("Downloading MovieLens-1M...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        print("Download complete.")
    else:
        print("Dataset already exists, skipping download.")


def load_movielens(data_dir: str = "data"):
    ratings = pd.read_csv(
        os.path.join(data_dir, "ml-1m", "ratings.dat"),
        sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )
    movies = pd.read_csv(
        os.path.join(data_dir, "ml-1m", "movies.dat"),
        sep="::", engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1"
    )
    return ratings, movies


def build_content_features(movies: pd.DataFrame):
    movies = movies.copy()
    movies["genre_list"] = movies["genres"].str.split("|")

    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies["genre_list"]).astype(np.float32)

    movies["year"] = movies["title"].str.extract(r'\((\d{4})\)').astype(float)
    movies["year"] = movies["year"].fillna(movies["year"].median())

    year_min = movies["year"].min()
    year_max = movies["year"].max()
    year_norm = ((movies["year"] - year_min) / (year_max - year_min + 1e-8)
                 ).values.reshape(-1, 1).astype(np.float32)

    features = np.hstack([genre_matrix, year_norm])

    return features, list(movies["movie_id"]), list(mlb.classes_) + ["year"]


def build_interactions(ratings: pd.DataFrame, pos_threshold: int = 4):
    pos = ratings[ratings["rating"] >= pos_threshold]

    all_users  = sorted(ratings["user_id"].unique())
    all_movies = sorted(ratings["movie_id"].unique())

    user2idx  = {u: i for i, u in enumerate(all_users)}
    movie2idx = {m: i for i, m in enumerate(all_movies)}
    idx2movie = {i: m for m, i in movie2idx.items()}

    user_pos_items = defaultdict(set)
    for _, row in pos.iterrows():
        user_pos_items[user2idx[row["user_id"]]].add(movie2idx[row["movie_id"]])

    return dict(user_pos_items), user2idx, movie2idx, idx2movie


def build_collab_embeddings(user_pos_items: dict, n_users: int,
                             n_movies: int, emb_dim: int = 32):
    rows, cols, data = [], [], []
    for u, items in user_pos_items.items():
        for i in items:
            rows.append(u); cols.append(i); data.append(1.0)

    R = csr_matrix((data, (rows, cols)), shape=(n_users, n_movies))
    _, sigma, Vt = svds(R.astype(np.float32), k=emb_dim)

    item_emb = Vt.T
    norms    = np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-8
    return (item_emb / norms).astype(np.float32)


def split_cold_warm(user_pos_items: dict, n_movies: int,
                    cold_threshold: int = 5):
    item_count = defaultdict(int)
    for items in user_pos_items.values():
        for item in items:
            item_count[item] += 1

    warm = [i for i in range(n_movies) if item_count[i] >= cold_threshold]
    cold = [i for i in range(n_movies) if item_count[i] <  cold_threshold]
    return warm, cold
