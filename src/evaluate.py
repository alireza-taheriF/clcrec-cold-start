from __future__ import annotations

import numpy as np


def get_user_embedding(user_idx: int, item_emb: np.ndarray,
                       user_pos_items: dict, exclude: set = None):
    pos = user_pos_items.get(user_idx, set())
    train = pos - (exclude or set())
    if not train:
        return None
    u_emb = item_emb[list(train)].mean(axis=0)
    norm = np.linalg.norm(u_emb)
    return u_emb / norm if norm > 1e-8 else None


def _rank_one(u_emb, item_emb, gt, candidates, Ks):
    scores = item_emb[candidates] @ u_emb
    order = np.argsort(-scores)
    ranked = candidates[order]
    out = {}
    for K in Ks:
        top = ranked[:K]
        hit = set(top.tolist()) & gt
        hr = float(len(hit) > 0)
        recall = len(hit) / len(gt) if gt else 0.0
        dcg = sum(1.0 / np.log2(r + 2)
                  for r, it in enumerate(top) if it in gt)
        ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt), K)))
        ndcg = dcg / ideal if ideal > 0 else 0.0
        out[K] = {"HR": hr, "Recall": recall, "NDCG": ndcg}
    return out


def hit_rate_and_ndcg(user_idx: int, item_emb: np.ndarray,
                      user_pos_items: dict, eval_item_pool: np.ndarray,
                      K: int = 10):
    pos = user_pos_items.get(user_idx, set())
    gt = pos & set(eval_item_pool.tolist())
    if not gt:
        return None, None

    u_emb = get_user_embedding(user_idx, item_emb, user_pos_items, exclude=gt)
    if u_emb is None:
        return None, None

    one = _rank_one(u_emb, item_emb, gt, eval_item_pool, [K])[K]
    return one["HR"], one["NDCG"]


def evaluate(item_emb: np.ndarray, user_pos_items: dict,
             test_users: list, cold_items: list,
             Ks: list = None):
    """Legacy cold-pool protocol (reproduces the original README table)."""
    if Ks is None:
        Ks = [5, 10, 20]
    pool = np.array(cold_items)
    per_k = {K: {"HR": [], "NDCG": []} for K in Ks}

    for u in test_users:
        pos = user_pos_items.get(u, set())
        gt = pos & set(pool.tolist())
        if not gt:
            continue
        u_emb = get_user_embedding(u, item_emb, user_pos_items, exclude=gt)
        if u_emb is None:
            continue
        metrics = _rank_one(u_emb, item_emb, gt, pool, Ks)
        for K in Ks:
            per_k[K]["HR"].append(metrics[K]["HR"])
            per_k[K]["NDCG"].append(metrics[K]["NDCG"])

    results = {}
    print(f"\n{'K':>4} | {'HR@K':>8} | {'NDCG@K':>8} | {'#Users':>7}")
    print("-" * 36)
    for K in Ks:
        hrs, ndcgs = per_k[K]["HR"], per_k[K]["NDCG"]
        results[K] = {
            "HR": float(np.mean(hrs)) if hrs else 0.0,
            "NDCG": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "n": len(hrs),
        }
        print(f"{K:>4} | {results[K]['HR']:>8.4f} | {results[K]['NDCG']:>8.4f} | {results[K]['n']:>7}")
    return results


def bootstrap_ci(values, n_boot: int = 200, seed: int = 42):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    mean = float(values.mean())
    if len(values) < 8:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boots[i] = rng.choice(values, size=len(values), replace=True).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi)


def evaluate_academic(item_emb: np.ndarray,
                      user_history: dict,
                      user_truth: dict,
                      test_users: list,
                      candidate_items,
                      Ks: list = None,
                      n_bootstrap: int = 200,
                      seed: int = 42,
                      title: str = ""):
    """
    Academic protocol: user embedding from warm history only;
    rank over the given candidate pool (cold / warm / all minus history).
    """
    if Ks is None:
        Ks = [5, 10, 20]
    cand_all = np.asarray(candidate_items)
    cand_set = set(cand_all.tolist())

    bags = {K: {"HR": [], "Recall": [], "NDCG": []} for K in Ks}

    for u in test_users:
        hist = user_history.get(u, set())
        gt = user_truth.get(u, set()) & cand_set
        if not hist or not gt:
            continue
        u_emb = item_emb[list(hist)].mean(axis=0)
        norm = np.linalg.norm(u_emb)
        if norm < 1e-8:
            continue
        u_emb = u_emb / norm
        candidates = np.array([i for i in cand_all if i not in hist], dtype=int)
        if len(candidates) == 0:
            continue
        metrics = _rank_one(u_emb, item_emb, gt, candidates, Ks)
        for K in Ks:
            for key in ("HR", "Recall", "NDCG"):
                bags[K][key].append(metrics[K][key])

    results = {}
    n_users = len(bags[Ks[0]]["HR"])
    if title:
        print(f"\n── {title} ──  users={n_users}")
    print(f"{'K':>4} | {'HR':>14} | {'Recall':>14} | {'NDCG':>14}")
    print("-" * 56)
    for K in Ks:
        results[K] = {"n": n_users}
        cells = []
        for key in ("HR", "Recall", "NDCG"):
            mean, lo, hi = bootstrap_ci(bags[K][key], n_bootstrap, seed)
            results[K][key] = mean
            results[K][f"{key}_lo"] = lo
            results[K][f"{key}_hi"] = hi
            cells.append(f"{mean:.4f}±{(hi-lo)/2:.3f}")
        print(f"{K:>4} | {cells[0]:>14} | {cells[1]:>14} | {cells[2]:>14}")
    return results


def build_user_splits(user_pos_items: dict, warm_items, cold_items):
    warm = set(warm_items)
    cold = set(cold_items)
    history, truth_cold, truth_warm = {}, {}, {}
    for u, items in user_pos_items.items():
        history[u] = items & warm
        truth_cold[u] = items & cold
        truth_warm[u] = items & warm
    return history, truth_cold, truth_warm
