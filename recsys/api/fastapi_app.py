"""FastAPI service exposing cold-start recommendations.

The API is deliberately thin: all the heavy lifting (training, SVD, encoding)
happens offline in ``main.py``, which writes an embedding table + metadata to
``results/artifacts/``. This app loads those artifacts once at startup and
answers requests by cosine similarity in the shared embedding space.

Run with::

    python -m recsys.api.fastapi_app        # uvicorn on 127.0.0.1:8000
    bash scripts/run_api.sh

Note: this service has no authentication. It is intended for local use only.
Add auth (and rate limiting) before exposing it on a network.
"""
from __future__ import annotations

import os
from typing import List, Optional

from pydantic import BaseModel, Field

from config import Config
from recsys.data.artifacts import ArtifactStore

try:
    from fastapi import FastAPI, HTTPException
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "FastAPI is required to run the API. Install with "
        "`pip install fastapi uvicorn`."
    ) from exc


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #
class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="Internal user index to recommend for.")
    top_k: int = Field(10, ge=1, le=100, description="Number of recommendations.")


class Explanation(BaseModel):
    score: float = Field(..., description="Cosine similarity score.")
    matched_genres: List[str] = Field(default_factory=list, description="Genres shared with user history.")
    most_similar_liked_movie: Optional[str] = Field(None, description="Most similar movie in user history.")
    reason: str = Field(..., description="Human-readable explanation of recommendation.")


class RecommendedItem(BaseModel):
    item_index: int
    movie_id: Optional[int] = None
    title: Optional[str] = None
    genres: Optional[str] = None
    explanation: Optional[Explanation] = None


class RecommendResponse(BaseModel):
    user_id: int
    backend: str
    count: int
    recommendations: List[RecommendedItem]


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(config: Optional[Config] = None) -> "FastAPI":
    """Build the FastAPI app, loading artifacts lazily on first use.

    Artifacts are loaded on demand (not at import time) so the module can be
    imported cheaply and so a missing-artifacts state produces a clean 503
    rather than a crash at startup.
    """
    config = config or Config()
    config.paths.ensure()
    app = FastAPI(title="CLCRec Cold-Start Recommender", version="1.0.0")

    state: dict = {"store": None}

    def get_store() -> ArtifactStore:
        if state["store"] is None:
            npz, meta = config.paths.item_embeddings, config.paths.metadata
            if not (os.path.exists(npz) and os.path.exists(meta)):
                raise HTTPException(
                    status_code=503,
                    detail="Model artifacts not found. Train first: "
                           "`python main.py --backend numpy --save-artifacts`.",
                )
            state["store"] = ArtifactStore.load(npz, meta)
        return state["store"]

    @app.get("/health")
    def health() -> dict:
        """Liveness probe. Reports whether artifacts are loaded and the backend."""
        loaded = state["store"] is not None or (
            os.path.exists(config.paths.item_embeddings)
            and os.path.exists(config.paths.metadata)
        )
        backend = None
        if loaded:
            try:
                backend = get_store().backend
            except HTTPException:
                loaded = False
        return {"status": "ok", "artifacts_available": loaded, "backend": backend}

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(req: RecommendRequest) -> RecommendResponse:
        """Return top-``k`` cold-start recommendations for a known user."""
        store = get_store()
        if not store.known_user(req.user_id):
            raise HTTPException(
                status_code=404,
                detail=f"Unknown user_id {req.user_id}.",
            )
        recs = store.recommend(req.user_id, req.top_k)
        items = [RecommendedItem(**store.item_meta(idx, user_id=req.user_id, score=score))
                 for idx, score in recs]
        return RecommendResponse(
            user_id=req.user_id,
            backend=store.backend,
            count=len(items),
            recommendations=items,
        )

    return app


# Module-level app so `uvicorn recsys.api.fastapi_app:app` works.
app = create_app()


def main() -> None:
    """Entry point for `python -m recsys.api.fastapi_app`."""
    import uvicorn

    host = os.environ.get("RECSYS_API_HOST", "127.0.0.1")
    port = int(os.environ.get("RECSYS_API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
