# Metrics Summary

Evaluated on cold items (MovieLens-1M). Higher is better.

| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|---|
| CLCRec (Contrastive) | 0.0469 | 0.0938 | 0.1406 | 0.0083 | 0.0230 | 0.0319 |
| Content-KNN | 0.0156 | 0.0156 | 0.1250 | 0.0032 | 0.0053 | 0.0256 |
| SVD-CF | 0.4688 | 0.5625 | 0.7031 | 0.2551 | 0.2758 | 0.3235 |

## Configuration

- **backend**: torch
- **emb_dim**: 32
- **hidden_dim**: 32
- **torch_hidden_dim**: 64
- **epochs**: 60
- **batch_size**: 256
- **lr**: 0.005
- **tau**: 0.1
- **cold_threshold**: 10
- **n_test_users**: 300
- **seed**: 42
