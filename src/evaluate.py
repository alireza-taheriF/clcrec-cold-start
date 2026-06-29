import numpy as np


def get_user_embedding(user_idx: int, item_emb: np.ndarray,
                       user_pos_items: dict, exclude: set = None):
    pos    = user_pos_items.get(user_idx, set())
    train  = pos - (exclude or set())
    if not train:
        return None
    u_emb = item_emb[list(train)].mean(axis=0)
    norm  = np.linalg.norm(u_emb)
    return u_emb / norm if norm > 1e-8 else None


def hit_rate_and_ndcg(user_idx: int, item_emb: np.ndarray,
                      user_pos_items: dict, eval_item_pool: np.ndarray,
                      K: int = 10):
    pos = user_pos_items.get(user_idx, set())
    gt  = pos & set(eval_item_pool.tolist())
    if not gt:
        return None, None

    u_emb = get_user_embedding(user_idx, item_emb, user_pos_items, exclude=gt)
    if u_emb is None:
        return None, None

    scores    = item_emb[eval_item_pool] @ u_emb
    top_local = np.argsort(scores)[::-1][:K]
    top_items = set(eval_item_pool[top_local].tolist())

    hr = float(len(top_items & gt) > 0)

    dcg      = sum(1.0 / np.log2(r + 2)
                   for r, it in enumerate(eval_item_pool[top_local])
                   if it in gt)
    ideal    = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt), K)))
    ndcg     = dcg / ideal if ideal > 0 else 0.0

    return hr, ndcg


def evaluate(item_emb: np.ndarray, user_pos_items: dict,
             test_users: list, cold_items: list,
             Ks: list = [5, 10, 20]):
    pool = np.array(cold_items)
    results = {}

    for K in Ks:
        hrs, ndcgs = [], []
        for u in test_users:
            hr, ndcg = hit_rate_and_ndcg(u, item_emb, user_pos_items, pool, K)
            if hr is not None:
                hrs.append(hr)
                ndcgs.append(ndcg)

        results[K] = {
            "HR":   float(np.mean(hrs))   if hrs   else 0.0,
            "NDCG": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "n":    len(hrs)
        }

    print(f"\n{'K':>4} | {'HR@K':>8} | {'NDCG@K':>8} | {'#Users':>7}")
    print("-" * 36)
    for K, res in results.items():
        print(f"{K:>4} | {res['HR']:>8.4f} | {res['NDCG']:>8.4f} | {res['n']:>7}")

    return results
