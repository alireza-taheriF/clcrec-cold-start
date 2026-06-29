# CLCRec Cold-Start

![Python](https://img.shields.io/badge/Python-3.12-blue)
![NumPy](https://img.shields.io/badge/NumPy-only-green)
![Dataset](https://img.shields.io/badge/Dataset-MovieLens--1M-orange)

A lightweight implementation of CLCRec (Contrastive Learning for Cold-Start Recommendation) on the MovieLens-1M dataset. The goal is to learn content-based item embeddings for cold items: items with insufficient user interactions and unreliable collaborative embeddings.

The model aligns content features (genre and release year) with SVD collaborative embeddings so that, at inference time, recommendations can be made in the same embedding space using content alone.

---

## Results

Evaluated on 215 users with cold item interactions (MovieLens-1M).

| Model    | HR@5   | HR@10  | HR@20  | NDCG@10 |
|----------|--------|--------|--------|---------|
| Baseline | 0.0186 | 0.0651 | 0.1256 | 0.0228  |
| CLCRec   | 0.0837 | 0.1302 | 0.1953 | 0.0474  |

CLCRec achieves **+100% HR@10** and **+108% NDCG@10** over the baseline.

---

## Overview

In classical recommender systems, item embeddings are learned from user-item interactions. For new or low-interaction (cold) items, these embeddings are weak or unavailable.

This project follows these steps:

1. Extract collaborative item embeddings from the interaction matrix using SVD.
2. Train a Content Encoder (MLP) to map content features into the same embedding space.
3. Optimize with InfoNCE: the content embedding of each warm item should be close to its collaborative embedding.
4. After training, encode all items (including cold ones) using content features only.
5. Generate recommendations via the mean embedding of a user's liked items and cosine similarity.

---

## Architecture

```
Content Features (genre + year)
        |
        v
  [Linear + ReLU]          Collaborative Embeddings (SVD)
        |                            |
        v                            |
  [Linear + L2-Norm]                 |
        |                            |
        +-------- InfoNCE Loss -------+
```

### Content Encoder

- Input: multi-hot genre vector + one normalized year dimension
- Layer 1: `content_dim -> hidden_dim` with ReLU
- Layer 2: `hidden_dim -> emb_dim`
- Output: L2-normalized embedding
- Optimizer: Adam (hand-written in NumPy)
- Regularization: L2 on weights

### InfoNCE Loss

In each batch of warm items, the positive pair is `(z_content[i], z_collab[i])`. All other pairs in the batch act as negatives. The `tau` (temperature) parameter controls the sharpness of the softmax.

---

## Project Structure

```
clcrec-cold-start/
├── main.py              # Entry point: full pipeline
├── requirements.txt     # Dependencies
├── results.png          # Output chart (generated after run)
├── data/                # MovieLens-1M (downloaded automatically)
└── src/
    ├── dataset.py       # Download, preprocessing, SVD, cold/warm split
    ├── model.py         # ContentEncoder and InfoNCE
    ├── train.py         # Training loop
    └── evaluate.py      # HR@K and NDCG@K
```

---

## Requirements

- Python 3.12 (recommended; Python 3.14 on macOS may have pip issues)
- Internet access to download MovieLens-1M (~5 MB)

---

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Core model and evaluation computations |
| pandas | Loading MovieLens data |
| scikit-learn | MultiLabelBinarizer for genres |
| scipy | SVD on sparse interaction matrix |
| matplotlib | Plotting results |
| tqdm | Imported in train; available for progress bars |

---

## Usage

```bash
source .venv/bin/activate
python main.py
```

### Pipeline Steps

1. Download MovieLens-1M to `data/ml-1m/` (if not present)
2. Build content features (genre + year)
3. Build interaction matrix and collaborative embeddings via SVD
4. Split items into warm/cold based on interaction count
5. Train the Content Encoder on warm items
6. Encode all items
7. Evaluate on cold items for 1,000 random users
8. Compare against a baseline (raw L2-normalized features, no training)
9. Save `results.png`

---

## Hyperparameters

Default settings in `main.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `EMB_DIM` | 32 | Embedding dimension (SVD and encoder) |
| `HIDDEN_DIM` | 32 | MLP hidden layer size |
| `COLD_THRESHOLD` | 10 | Items with fewer interactions are cold |
| `EPOCHS` | 150 | Number of training epochs |
| `BATCH_SIZE` | 256 | Batch size |
| `LR` | 0.005 | Adam learning rate |
| `TAU` | 0.1 | InfoNCE temperature |
| `L2_REG` | 5e-4 | L2 regularization coefficient |
| `N_TEST_USERS` | 1000 | Number of users for evaluation |
| `SEED` | 42 | Random seed |

Edit these values in `main.py` to change model behavior.

---

## Data

### MovieLens-1M

- Source: [GroupLens MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
- Files: `ratings.dat`, `movies.dat`, `users.dat`
- Positive interaction: rating >= 4

### Content Features

- **Genre**: multi-hot encoding over 18 MovieLens genres
- **Year**: extracted from title via regex `(YYYY)`, e.g. `"Toy Story (1995)"`
- Missing years are filled with the median and normalized to [0, 1]
- Final shape: `(n_movies, 19)` = 18 genres + 1 year

### Warm / Cold Split

- **Warm**: items with at least `COLD_THRESHOLD` positive interactions
- **Cold**: all other items
- Training uses warm items only; evaluation uses cold items

---

## Evaluation

### Protocol

- For each test user, user embedding = mean of liked item embeddings (excluding ground truth)
- Candidate pool = cold items only
- Ground truth = cold items the user has liked

### Metrics

| Metric | Meaning |
|--------|---------|
| **HR@K** (Hit Rate) | Whether at least one relevant item appears in top-K |
| **NDCG@K** | Ranking quality with higher weight on top positions |

Default K values: 5, 10, 20

### Baseline

Raw features (genre + year) are L2-normalized without training, then evaluated with the same protocol.

---

## Output

### Terminal

```
Users: 6040 | Movies: 3706 | Genres: 19
Warm: ... | Cold: ...
Training CLCRec | epochs=150 | batch=256 | τ=0.1
  Epoch  10/150  |  Loss: ...
  ...

── CLCRec (Contrastive) ──
   K |   HR@K | NDCG@K | #Users
  ...

── Baseline (Raw Genre Vectors) ──
   K |   HR@K | NDCG@K | #Users
  ...

Chart saved to results.png
```

### `results.png`

Two plots:

1. Hit Rate@K for CLCRec vs. Baseline
2. Training loss curve over epochs

---

## Modules

### `src/dataset.py`

| Function | Description |
|----------|-------------|
| `download_movielens` | Download and extract MovieLens-1M |
| `load_movielens` | Load ratings and movies |
| `build_content_features` | Build feature matrix (genre + year) |
| `build_interactions` | Build user-item positives and index mappings |
| `build_collab_embeddings` | SVD on sparse interaction matrix |
| `split_cold_warm` | Split items into warm/cold |

### `src/model.py`

| Class / Function | Description |
|------------------|-------------|
| `ContentEncoder` | Two-layer MLP with forward/backward/adam |
| `infonce_loss` | Compute loss and gradient w.r.t. z_content |

### `src/train.py`

| Function | Description |
|----------|-------------|
| `train` | Epoch/batch loop, shuffle warm items, return encoder and history |

### `src/evaluate.py`

| Function | Description |
|----------|-------------|
| `get_user_embedding` | Mean embedding of liked items |
| `hit_rate_and_ndcg` | HR@K and NDCG@K for one user |
| `evaluate` | Evaluate on test users and print results table |

---

## Implementation Notes

- **Pure NumPy**: forward pass, backward pass, and Adam are implemented manually; no autograd.
- **He initialization**: weights are initialized with `sqrt(2 / (fan_in + fan_out))`.
- **Small batches**: batches with fewer than 8 items are skipped.
- **Indexing**: `content_matrix` is ordered by `movie_id`; remapped to `movie_idx` in `main.py`.

---

## Limitations

- Only simple side information (genre + year) is used; title text or plot summaries are not used.
- Collaborative embeddings are fixed (SVD runs once) and not updated during training.
- Evaluation is not leave-one-out; all cold items liked by a user are in the candidate pool.
- Results are sensitive to seed and hyperparameters.

---

## References

- MovieLens: Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets.
- InfoNCE / Contrastive Learning: Oord, A. van den, et al. (2018). Representation Learning with Contrastive Predictive Coding.
- CLCRec: Wei, W., et al. (2021). Contrastive Learning for Cold-Start Recommendation.
