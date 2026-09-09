"""FirstSlot: allocate the first impressions of a zero-sale item."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def user_vectors(item_emb: np.ndarray, user_pos_items: dict, warm: set) -> dict:
    out = {}
    for u, items in user_pos_items.items():
        hist = [i for i in items if i in warm]
        if not hist:
            continue
        vec = item_emb[hist].mean(axis=0)
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            continue
        out[int(u)] = (vec / norm).astype(np.float32)
    return out


def affinities(query: np.ndarray, user_embs: dict) -> dict:
    return {u: float(vec @ query) for u, vec in user_embs.items()}


def allocate(user_scores: dict, segments: dict, budget: int, diversity: float = 0.15):
    """Greedy: affinity minus a penalty for repeating the same plant segment."""
    remaining = [u for u, s in user_scores.items() if np.isfinite(s)]
    remaining.sort(key=lambda u: user_scores[u], reverse=True)
    selected = []
    counts = Counter()
    for _ in range(min(budget, len(remaining))):
        best, best_s = None, -1e9
        for u in remaining:
            seg = segments.get(u, "unknown")
            score = user_scores[u] - diversity * counts[seg]
            if score > best_s:
                best, best_s = u, score
        selected.append(best)
        remaining.remove(best)
        counts[segments.get(best, "unknown")] += 1
    return selected


def popular_users(user_pos_items: dict, budget: int):
    counts = Counter({u: len(items) for u, items in user_pos_items.items()})
    return [u for u, _ in counts.most_common(budget)]


def category_buyers(user_pos_items: dict, item_category: dict, category: str, budget: int):
    counts = Counter()
    for u, items in user_pos_items.items():
        n = sum(1 for i in items if item_category.get(i) == category)
        if n:
            counts[u] = n
    return [u for u, _ in counts.most_common(budget)]


def precision_recall(selected, truth_users):
    if not selected:
        return 0.0, 0.0
    hit = set(selected) & set(truth_users)
    prec = len(hit) / len(selected)
    rec = len(hit) / len(truth_users) if truth_users else 0.0
    return prec, rec


def policy_compare(item_emb, user_pos_items, warm, cold, truth_by_item,
                   item_category, user_segments, budgets=(20, 40, 80),
                   diversity=0.35, seed=0):
    """Offline launch KPI: of the first B impressions, how many hit real buyers."""
    user_embs = user_vectors(item_emb, user_pos_items, set(warm))
    rng = np.random.default_rng(seed)
    all_users = list(user_embs.keys())
    rows = []

    eligible = []
    for item in cold:
        truth = truth_by_item.get(int(item), set())
        if len(truth) >= 3:
            eligible.append(int(item))

    for B in budgets:
        bags = defaultdict(lambda: {"prec": [], "rec": []})
        for item in eligible:
            query = item_emb[item]
            qn = np.linalg.norm(query)
            if qn < 1e-8:
                continue
            query = query / qn
            truth = truth_by_item[item]
            aff = affinities(query, user_embs)

            plans = {
                "random": list(rng.choice(all_users, size=min(B, len(all_users)), replace=False)),
                "popular": popular_users(user_pos_items, B),
                "category": category_buyers(user_pos_items, item_category, item_category.get(item), B),
                "firstslot": allocate(aff, user_segments, B, diversity),
            }
            for name, chosen in plans.items():
                prec, rec = precision_recall(chosen, truth)
                bags[name]["prec"].append(prec)
                bags[name]["rec"].append(rec)

        for name, vals in bags.items():
            rows.append({
                "budget": B,
                "policy": name,
                "precision": float(np.mean(vals["prec"])) if vals["prec"] else 0.0,
                "recall": float(np.mean(vals["rec"])) if vals["rec"] else 0.0,
                "n_items": len(vals["prec"]),
            })
    return rows, eligible


def explain_user(user_idx, item_query, item_emb, liked_idx, idx2item, meta, k=3):
    if not liked_idx:
        return []
    liked = list(liked_idx)
    scores = item_emb[liked] @ item_query
    top = np.argsort(-scores)[:k]
    out = []
    for local in top:
        idx = int(liked[local])
        item_id = idx2item[idx]
        info = meta.get(str(item_id)) or meta.get(item_id) or {}
        out.append({
            "item_id": int(item_id),
            "title": info.get("title", ""),
            "score": float(scores[local]),
        })
    return out
