import unittest

from src.dataset import build_content_features, build_interactions
from src.launch import allocate, precision_recall
from src.mro import generate_mro, to_movies_frame, to_ratings_frame
from src.product import run_firstslot
from src.recommend import ColdStartRecommender


class MroTests(unittest.TestCase):
    def test_cold_items_are_in_catalog_not_in_train_events(self):
        world = generate_mro(seed=1, plants_per_industry=4)
        cold_ids = set(world.items.loc[world.items["is_cold"] == 1, "item_id"])
        self.assertGreater(len(cold_ids), 5)
        self.assertTrue(cold_ids.isdisjoint(set(world.events["item_id"])))
        self.assertTrue(cold_ids.issubset(set(world.truth["item_id"])))

    def test_content_features_are_finite(self):
        world = generate_mro(seed=1, plants_per_industry=4)
        X, _, names, _ = build_content_features(to_movies_frame(world.items))
        self.assertFalse((X != X).any())
        self.assertGreater(len(names), 10)

    def test_mappings_include_zero_sale_skus(self):
        world = generate_mro(seed=1, plants_per_industry=4)
        ratings = to_ratings_frame(world.events)
        _, _, movie2idx, _ = build_interactions(
            ratings, pos_threshold=1,
            all_item_ids=world.items["item_id"].tolist(),
            all_user_ids=world.users["user_id"].tolist(),
        )
        cold_ids = world.items.loc[world.items["is_cold"] == 1, "item_id"]
        self.assertTrue(all(int(i) in movie2idx for i in cold_ids))


class AllocationTests(unittest.TestCase):
    def test_diversity_spreads_segments(self):
        scores = {i: 1.0 - i * 0.01 for i in range(12)}
        segs = {i: "A" if i < 8 else "B" for i in range(12)}
        greedy = [u for u, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:6]]
        diverse = allocate(scores, segs, 6, diversity=0.8)
        self.assertEqual(len(diverse), 6)
        self.assertGreater(sum(1 for u in diverse if segs[u] == "B"),
                           sum(1 for u in greedy if segs[u] == "B"))

    def test_precision_recall(self):
        prec, rec = precision_recall([1, 2, 3, 4], {2, 4, 8})
        self.assertAlmostEqual(prec, 0.5)
        self.assertAlmostEqual(rec, 2 / 3)


class ProductSmoke(unittest.TestCase):
    def test_firstslot_trains_and_plans(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            report = run_firstslot(td, epochs=4, seed=3)
            self.assertEqual(report["product"], "FirstSlot")
            rec = ColdStartRecommender.load(td)
            items = rec.list_new_items()
            self.assertGreater(len(items), 0)
            plan = rec.plan_launch(item_id=items[0]["item_id"], budget=15, diversity=0.3)
            self.assertGreaterEqual(len(plan["audience"]), 5)
            self.assertTrue(plan["audience"][0]["name"])
            fresh = rec.plan_launch(
                title="بلبرینگ 6205-2RS Nachi",
                tags="bearing|Nachi|6205-2RS|ISO|automotive|steel",
                budget=10,
            )
            self.assertEqual(len(fresh["audience"]), 10)


if __name__ == "__main__":
    unittest.main()
