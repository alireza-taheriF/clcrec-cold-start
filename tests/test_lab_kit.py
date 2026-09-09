import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.baselines import ridge_content_map
from src.dataset import (
    ContentFeaturizer,
    build_collab_embeddings,
    mask_cold_interactions,
)
from src.evaluate import evaluate_academic
from src.model import ContentEncoder, infonce_loss
from src.recommend import ColdStartRecommender
from src.train import train


class EncoderTests(unittest.TestCase):
    def test_forward_normalized(self):
        enc = ContentEncoder(6, 4, 8, lr=0.01)
        z, _ = enc.forward(np.random.randn(5, 6).astype(np.float32))
        norms = np.linalg.norm(z, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))

    def test_infonce_finite(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(16, 8)).astype(np.float32)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b = rng.normal(size=(16, 8)).astype(np.float32)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        loss, grad = infonce_loss(a, b, tau=0.1)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(grad.shape, a.shape)

    def test_save_load_roundtrip(self):
        enc = ContentEncoder(5, 3, 7, lr=0.01)
        x = np.random.randn(4, 5).astype(np.float32)
        z1, _ = enc.forward(x)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "enc.npz")
            enc.save(path)
            enc2 = ContentEncoder.load(path)
        z2, _ = enc2.forward(x)
        self.assertTrue(np.allclose(z1, z2, atol=1e-6))


class LeakageTests(unittest.TestCase):
    def test_cold_items_excluded_from_svd(self):
        user_pos = {
            0: {0, 1, 2},
            1: {0, 1, 2},
            2: {0, 1},
        }
        emb = build_collab_embeddings(user_pos, n_users=3, n_movies=3,
                                      emb_dim=2, exclude_items=[2])
        self.assertEqual(emb.shape, (3, 2))
        self.assertLess(float(np.linalg.norm(emb[2])), 1e-5)

    def test_mask_cold_interactions(self):
        masked = mask_cold_interactions({0: {1, 2, 3}}, [2, 3])
        self.assertEqual(masked[0], {1})


class FeaturizerTests(unittest.TestCase):
    def test_new_item_has_same_dim(self):
        movies = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "title": ["Toy Story (1995)", "Jumanji (1995)", "Heat (1995)"],
            "genres": ["Animation|Comedy", "Adventure|Children", "Action|Crime"],
        })
        feat = ContentFeaturizer(use_title=True, title_svd_dim=2)
        X, _, names = feat.fit_transform(movies)
        z = feat.transform_one("Dune", "Action|Adventure|Sci-Fi", 2021)
        self.assertEqual(X.shape[1], len(names))
        self.assertEqual(z.shape[0], X.shape[1])


class TrainEvalTests(unittest.TestCase):
    def test_short_train_and_academic_eval(self):
        rng = np.random.default_rng(1)
        n_items, n_users, d = 40, 12, 6
        content = rng.normal(size=(n_items, 5)).astype(np.float32)
        user_pos = {u: set(rng.choice(n_items, size=8, replace=False).tolist())
                    for u in range(n_users)}
        warm = list(range(30))
        cold = list(range(30, 40))
        collab = build_collab_embeddings(user_pos, n_users, n_items, d,
                                         exclude_items=cold)
        enc, history = train(
            content, collab, warm, emb_dim=d, hidden_dim=8,
            epochs=3, batch_size=16, lr=0.05, tau=0.2, l2_reg=1e-4,
            seed=0, patience=5, val_warm_frac=0.2,
        )
        self.assertGreaterEqual(len(history), 1)
        emb = enc.encode_all(content)
        hist = {u: items & set(warm) for u, items in user_pos.items()}
        truth = {u: items & set(cold) for u, items in user_pos.items()}
        table = evaluate_academic(
            emb, hist, truth, list(range(n_users)), cold,
            Ks=[5, 10], n_bootstrap=20, seed=0, title="unit",
        )
        self.assertIn("Recall", table[5])
        self.assertGreaterEqual(table[5]["n"], 1)

        ridge = ridge_content_map(content, collab, warm)
        self.assertEqual(ridge.shape, emb.shape)


class RecommenderTests(unittest.TestCase):
    def test_save_load_and_new_item(self):
        movies = pd.DataFrame({
            "movie_id": [10, 20, 30, 40],
            "title": ["Alpha (1999)", "Beta (2001)", "Gamma (2003)", "Delta (2005)"],
            "genres": ["Action", "Action|Sci-Fi", "Drama", "Action|Drama"],
        })
        feat = ContentFeaturizer(use_title=True, title_svd_dim=2)
        X, _, _ = feat.fit_transform(movies)
        enc = ContentEncoder(X.shape[1], 4, 6, lr=0.01)
        for _ in range(5):
            z, cache = enc.forward(X)
            target = z.copy()
            loss, grad = infonce_loss(z, target, 0.2)
            grads = enc.backward(grad, cache, 1e-4)
            enc.adam_step(*grads)
        item_emb = enc.encode_all(X)
        rec = ColdStartRecommender(
            encoder=enc,
            item_emb=item_emb,
            featurizer=feat,
            idx2movie={0: 10, 1: 20, 2: 30, 3: 40},
            movie2idx={10: 0, 20: 1, 30: 2, 40: 3},
            meta={10: {"title": "Alpha (1999)", "genres": "Action"},
                  20: {"title": "Beta (2001)", "genres": "Action|Sci-Fi"},
                  30: {"title": "Gamma (2003)", "genres": "Drama"},
                  40: {"title": "Delta (2005)", "genres": "Action|Drama"}},
            warm_items=[0, 1, 2],
            cold_items=[3],
            user_pos_items={0: {0, 1}},
            user2idx={1: 0},
        )
        neighbors = rec.similar_items(10, k=2)
        self.assertEqual(len(neighbors), 2)
        self.assertNotEqual(neighbors[0].item_id, 10)
        fresh = rec.recommend_for_new_item("Epsilon", "Action|Sci-Fi", 2024, k=2)
        self.assertEqual(len(fresh), 2)

        with tempfile.TemporaryDirectory() as td:
            rec.save(td)
            rec2 = ColdStartRecommender.load(td)
            self.assertEqual(len(rec2.similar_items(10, k=2)), 2)
            self.assertTrue(os.path.exists(os.path.join(td, "encoder.npz")))
            with open(os.path.join(td, "catalog.json")) as f:
                catalog = json.load(f)
            self.assertIn("movie2idx", catalog)


if __name__ == "__main__":
    unittest.main()
