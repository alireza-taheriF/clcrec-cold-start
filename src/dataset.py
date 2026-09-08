from __future__ import annotations

import os
import pickle
import zipfile
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


class ContentFeaturizer:
    """Fits genre / year / optional title features and transforms new items."""

    def __init__(self, use_title: bool = True, title_svd_dim: int = 16):
        self.use_title = use_title
        self.title_svd_dim = title_svd_dim
        self.mlb = MultiLabelBinarizer()
        self.tfidf = None
        self.svd = None
        self.year_min = None
        self.year_max = None
        self.year_median = None
        self.feature_names: list[str] = []

    def _genre_matrix(self, genre_lists, fit: bool = False) -> np.ndarray:
        if fit:
            self.mlb.fit(genre_lists)
        index = {name: i for i, name in enumerate(self.mlb.classes_)}
        out = np.zeros((len(genre_lists), len(self.mlb.classes_)), dtype=np.float32)
        for row, genres in enumerate(genre_lists):
            for name in genres:
                col = index.get(name)
                if col is not None:
                    out[row, col] = 1.0
        return out

    def _titles_without_year(self, titles: pd.Series) -> pd.Series:
        return titles.fillna("").str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)

    def _year_column(self, movies: pd.DataFrame) -> np.ndarray:
        years = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
        if self.year_median is None:
            self.year_median = float(years.iloc[:, 0].median())
        years = years.fillna(self.year_median)
        year_values = years.iloc[:, 0].to_numpy(dtype=np.float32)
        if self.year_min is None:
            self.year_min = float(year_values.min())
            self.year_max = float(year_values.max())
        denom = (self.year_max - self.year_min) + 1e-8
        return ((year_values - self.year_min) / denom).reshape(-1, 1).astype(np.float32)

    def fit_transform(self, movies: pd.DataFrame):
        movies = movies.copy()
        movies["genre_list"] = movies["genres"].fillna("(no genres listed)").str.split("|")
        genre_matrix = self._genre_matrix(movies["genre_list"].tolist(), fit=True)
        year_norm = self._year_column(movies)

        parts = [genre_matrix, year_norm]
        names = list(self.mlb.classes_) + ["year"]

        if self.use_title:
            clean = self._titles_without_year(movies["title"])
            self.tfidf = TfidfVectorizer(
                max_features=max(self.title_svd_dim * 8, 64),
                ngram_range=(1, 2),
                min_df=1,
            )
            tfidf_matrix = self.tfidf.fit_transform(clean)
            n_comp = min(self.title_svd_dim, max(1, tfidf_matrix.shape[1] - 1),
                         max(1, tfidf_matrix.shape[0] - 1))
            self.svd = TruncatedSVD(n_components=n_comp, random_state=0)
            title_z = self.svd.fit_transform(tfidf_matrix).astype(np.float32)
            parts.append(title_z)
            names.extend([f"title_{i}" for i in range(title_z.shape[1])])

        features = np.hstack(parts).astype(np.float32)
        self.feature_names = names
        return features, list(movies["movie_id"]), names

    def transform(self, movies: pd.DataFrame) -> np.ndarray:
        movies = movies.copy()
        movies["genre_list"] = movies["genres"].fillna("(no genres listed)").str.split("|")
        genre_matrix = self._genre_matrix(movies["genre_list"].tolist(), fit=False)
        year_norm = self._year_column(movies)
        parts = [genre_matrix, year_norm]
        if self.use_title:
            clean = self._titles_without_year(movies["title"])
            title_z = self.svd.transform(self.tfidf.transform(clean)).astype(np.float32)
            parts.append(title_z)
        return np.hstack(parts).astype(np.float32)

    def transform_one(self, title: str, genres: str, year: float | None = None) -> np.ndarray:
        if year is not None and "(" not in (title or ""):
            title = f"{title} ({int(year)})"
        row = pd.DataFrame([{"movie_id": -1, "title": title, "genres": genres}])
        return self.transform(row)[0]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "ContentFeaturizer":
        with open(path, "rb") as f:
            return pickle.load(f)


def download_movielens(data_dir: str = "data"):
    os.makedirs(data_dir, exist_ok=True)
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
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
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        os.path.join(data_dir, "ml-1m", "movies.dat"),
        sep="::", engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    return ratings, movies


def build_content_features(movies: pd.DataFrame, use_title: bool = True,
                           title_svd_dim: int = 16):
    featurizer = ContentFeaturizer(use_title=use_title, title_svd_dim=title_svd_dim)
    features, movie_ids, names = featurizer.fit_transform(movies)
    return features, movie_ids, names, featurizer


def build_interactions(ratings: pd.DataFrame, pos_threshold: int = 4):
    pos = ratings[ratings["rating"] >= pos_threshold]

    all_users = sorted(ratings["user_id"].unique())
    all_movies = sorted(ratings["movie_id"].unique())

    user2idx = {u: i for i, u in enumerate(all_users)}
    movie2idx = {m: i for i, m in enumerate(all_movies)}
    idx2movie = {i: m for m, i in movie2idx.items()}

    user_pos_items = defaultdict(set)
    for _, row in pos.iterrows():
        user_pos_items[user2idx[row["user_id"]]].add(movie2idx[row["movie_id"]])

    return dict(user_pos_items), user2idx, movie2idx, idx2movie


def align_content_features(content_matrix, movie_id_list, movie2idx, n_features):
    n_movies = len(movie2idx)
    aligned = np.zeros((n_movies, n_features), dtype="float32")
    for raw_idx, movie_id in enumerate(movie_id_list):
        if movie_id in movie2idx:
            aligned[movie2idx[movie_id]] = content_matrix[raw_idx]
    return aligned


def build_collab_embeddings(user_pos_items: dict, n_users: int,
                            n_movies: int, emb_dim: int = 32,
                            exclude_items=None):
    """SVD collaborative embeddings. Cold items must be excluded to avoid leakage."""
    exclude = set(exclude_items or [])
    rows, cols, data = [], [], []
    for u, items in user_pos_items.items():
        for i in items:
            if i in exclude:
                continue
            rows.append(u)
            cols.append(i)
            data.append(1.0)

    if not data:
        raise ValueError("No interactions left after excluding cold items.")

    R = csr_matrix((data, (rows, cols)), shape=(n_users, n_movies))
    k = min(emb_dim, max(1, min(R.shape) - 1))
    _, _, Vt = svds(R.astype(np.float32), k=k)

    item_emb = Vt.T
    if item_emb.shape[1] < emb_dim:
        pad = np.zeros((n_movies, emb_dim - item_emb.shape[1]), dtype=np.float32)
        item_emb = np.hstack([item_emb, pad])

    if exclude:
        item_emb[list(exclude)] = 0.0

    norms = np.linalg.norm(item_emb, axis=1, keepdims=True)
    item_emb = np.divide(item_emb, norms, out=np.zeros_like(item_emb), where=norms > 1e-8)
    return item_emb.astype(np.float32)


def split_cold_warm(user_pos_items: dict, n_movies: int,
                    cold_threshold: int = 5):
    item_count = defaultdict(int)
    for items in user_pos_items.values():
        for item in items:
            item_count[item] += 1

    warm = [i for i in range(n_movies) if item_count[i] >= cold_threshold]
    cold = [i for i in range(n_movies) if item_count[i] < cold_threshold]
    return warm, cold


def split_cold_warm_random(n_movies: int, cold_ratio: float = 0.2, seed: int = 42):
    """Paper-style random item hold-out (complete cold-start)."""
    rng = np.random.default_rng(seed)
    items = np.arange(n_movies)
    rng.shuffle(items)
    n_cold = max(1, int(round(n_movies * cold_ratio)))
    cold = sorted(items[:n_cold].tolist())
    warm = sorted(items[n_cold:].tolist())
    return warm, cold


def mask_cold_interactions(user_pos_items: dict, cold_items) -> dict:
    cold = set(cold_items)
    return {u: (items - cold) for u, items in user_pos_items.items()}
