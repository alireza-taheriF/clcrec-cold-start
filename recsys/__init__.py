"""CLCRec cold-start recommender package.

Modules:
    data        MovieLens loading, feature engineering, SVD, warm/cold split
    models      NumPy and PyTorch content encoders (share a common interface)
    training    Training loops for each backend
    evaluation  HR@K / NDCG@K metrics and baselines
    api         FastAPI service for serving recommendations
"""

__version__ = "0.2.0"
