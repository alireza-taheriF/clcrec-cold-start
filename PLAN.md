# PLAN — CLCRec Cold-Start → Portfolio-Grade Project

This document tracks the transformation of the current single-directory NumPy
prototype into a clean, modular, multi-backend recommender project with an HTTP
API, richer evaluation, and portfolio-grade documentation.

The guiding constraint: **the hand-written NumPy implementation is the teaching
core and must keep working unchanged in behavior.** Everything else is built
around it.

---

## 1. Current State (as of this session)

### What the pipeline does (`main.py`)
1. Download / load MovieLens-1M (`ratings.dat`, `movies.dat`).
2. Build content features: 18-genre multi-hot + 1 normalized year → shape `(n_movies, 19)`.
3. Build positive interactions (rating ≥ 4) and remap user/movie ids to dense indices.
4. Build collaborative item embeddings via **SVD** on the sparse interaction matrix (L2-normalized, dim 32).
5. Split items into **warm** (≥ `COLD_THRESHOLD` positive interactions) and **cold**.
6. Train a content encoder on **warm** items with InfoNCE, aligning content embeddings to SVD embeddings.
7. Encode all items (incl. cold) from content alone.
8. Evaluate HR@K / NDCG@K on **cold** items for N random users, vs. a raw-feature baseline.
9. Save `results.png` (HR@K comparison + training-loss curve).

### Model architecture & loss (`src/model.py`)
- `ContentEncoder`: `content_dim → hidden_dim → emb_dim`, ReLU on hidden, **L2-normalized** output.
- Weights init with fan-avg scaling; **hand-written Adam** (per-parameter m/v state).
- `forward` returns a cache; `backward` implements the L2-normalization Jacobian by hand + L2 weight decay.
- `infonce_loss`: in-batch contrastive loss. Positive pair = `(z_content[i], z_collab[i])`; all other
  columns are negatives. Returns both loss and the analytic gradient w.r.t. `z_content`.

### Training (`src/train.py`)
- Shuffles warm items each epoch, mini-batches (skips batches < 8), forward → InfoNCE → backward → Adam.
- Returns the trained encoder and per-epoch loss history.

### Evaluation (`src/evaluate.py`)
- **Cold items** = candidate pool. **User embedding** = mean of the user's liked-item embeddings
  (excluding held-out ground-truth cold items), L2-normalized.
- **Ground truth** = cold items the user liked. Score = cosine (dot on normalized vectors).
- HR@K = at least one hit in top-K; NDCG@K = position-weighted gain / ideal.

### Known inconsistencies to fix along the way
- `README` mentions "215 users" in one place but default `N_TEST_USERS=1000`.
- `main.py` uses `HIDDEN_DIM=32` while `model.py`/`train.py` default to 64. Config will make this explicit.

---

## 2. Target Architecture

```
                         MovieLens-1M (ratings.dat, movies.dat)
                                        │
                          ┌─────────────┴─────────────┐
                          ▼                             ▼
                 content features               positive interactions
              (genres multi-hot + year)          (rating ≥ threshold)
                          │                             │
                          │                     ┌───────┴────────┐
                          │                     ▼                ▼
                          │              SVD collab emb     warm / cold split
                          │              (item vectors)     (by interaction count)
                          │                     │                │
                          │        align (InfoNCE)              │
                          ▼                     │                │
                 ┌──────────────────┐           │                │
                 │  Content Encoder │◄──────────┘                │
                 │  numpy  |  torch │  (train on WARM items)     │
                 └────────┬─────────┘                            │
                          │ encode ALL items (incl. COLD)        │
                          ▼                                       │
                  item embeddings ──────────────────────────────┤
                          │                                       ▼
                          │                              evaluate on COLD items
                          ▼                              (HR@K, NDCG@K)
                 user embedding = mean(liked items)      + baselines: content-KNN, SVD-CF
                          │
                          ▼
                    top-K recommend  ◄──── FastAPI  (/health, /recommend)
```

### Target layout
```
.
├── main.py                     # CLI entrypoint (--backend numpy|torch)
├── config.py                   # dataclass of hyperparameters + paths
├── recsys/
│   ├── __init__.py
│   ├── base.py                 # RecommenderModel Protocol
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # load, features, SVD, splits (moved from src/dataset.py)
│   │   └── utils.py            # user/item embedding + save/load artifacts
│   ├── models/
│   │   ├── __init__.py
│   │   ├── numpy_encoder.py    # current NumPy encoder + InfoNCE (moved, unchanged behavior)
│   │   └── torch_encoder.py    # PyTorch encoder + InfoNCE (new)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer_numpy.py    # NumPy training loop (moved)
│   │   └── trainer_torch.py    # PyTorch training loop (new)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py          # HR@K, NDCG@K + baselines + logging
│   └── api/
│       ├── __init__.py
│       └── fastapi_app.py      # /health, /recommend
├── scripts/
│   ├── run_numpy.sh
│   ├── run_torch.sh
│   └── run_api.sh
├── results/                    # metrics json/csv, plots, logs
├── docs/                       # architecture diagram, HR@K snapshot
├── src/                        # ORIGINAL prototype kept as-is (back-compat shims)
└── README.md
```

The legacy `src/` package is preserved so the original prototype still runs; the
new code lives under `recsys/` and reuses the same logic.

---

## 3. Work Items This Session

- [x] **Step 1** — Read code, write this PLAN.md.
- [ ] **Step 2** — Refactor into `recsys/` package + `config.py` + `RecommenderModel` Protocol.
      Move NumPy encoder/trainer without behavior change. Keep repo runnable.
- [ ] **Step 3** — PyTorch backend: `torch_encoder.py` (content_dim → 64 → 32, ReLU, L2-norm),
      InfoNCE with adjustable/learnable τ, GPU-if-available, `trainer_torch.py`. Toggle via `--backend`.
- [ ] **Step 4** — FastAPI app: `/health`, `POST /recommend`. Loads saved artifacts (item embeddings +
      metadata + interactions), builds user embedding, returns top-K with title/genres.
- [ ] **Step 5** — Evaluation: add **content-KNN** and **SVD-CF** baselines; unified evaluation harness;
      write metrics to `results/metrics.json` + `results/metrics.csv`; regenerate `results/metrics_summary.md`.
- [ ] **Step 6** — README overhaul (problem, overview, Mermaid diagram, features, dataset, metrics, how-to-run,
      architecture, roadmap) + `docs/` diagram + HR@K snapshot.

## 4. Non-Goals / Constraints
- No Docker, no deployment (design for it, don't build it).
- No new network dependencies at runtime beyond the existing MovieLens download (data already local).
- Keep hand-written NumPy gradients readable and well-commented — do not over-abstract.
- Keep the codebase buildable/runnable after each step.
