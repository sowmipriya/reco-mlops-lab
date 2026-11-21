"""Real-time recommendation API.

Request path for GET /reco?client_id=...:
    1. read pre-built features + real-time recent events from the online store (1 Redis round trip)
    2. merge histories (real-time first, most recent first)
    3. score with the champion model held in memory (loaded once from the MLflow registry)
    4. return top-k with strategy, model version and latency
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from mlflow import MlflowClient
from pydantic import BaseModel

from reco.config import get_settings
from reco.feature_store.online import OnlineFeatureStore
from reco.models.pyfunc import load_recommender
from reco.models.recommender import EmbeddingRecommender

logger = logging.getLogger("reco.api")


class RecoResponse(BaseModel):
    client_id: str | None = None
    product_ids: list[str]
    scores: list[float]
    strategy: str
    model_version: str
    latency_ms: float


class Event(BaseModel):
    client_id: str
    product_id: str
    event_type: Literal["view", "add_to_cart", "purchase"] = "view"


class ModelHolder:
    def __init__(self) -> None:
        self.model: EmbeddingRecommender | None = None
        self.version: str = "none"

    def load_champion(self) -> None:
        s = get_settings()
        client = MlflowClient(tracking_uri=s.mlflow_tracking_uri)
        mv = client.get_model_version_by_alias(s.model_name, s.champion_alias)
        import mlflow

        mlflow.set_tracking_uri(s.mlflow_tracking_uri)
        self.model = load_recommender(f"models:/{s.model_name}@{s.champion_alias}")
        self.version = mv.version
        logger.info("Loaded %s version %s", s.model_name, mv.version)


def merge_history(realtime: list[str], batch: list[str], limit: int = 20) -> list[str]:
    """Real-time events first, then pre-built history, deduplicated, order preserved."""
    seen: set[str] = set()
    merged = []
    for pid in [*realtime, *batch]:
        if pid not in seen:
            seen.add(pid)
            merged.append(pid)
    return merged[:limit]


def create_app(store: OnlineFeatureStore | None = None, holder: ModelHolder | None = None) -> FastAPI:
    s = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store or OnlineFeatureStore.from_url(s.redis_url, s.recent_events_max)
        app.state.holder = holder or ModelHolder()
        if app.state.holder.model is None:
            app.state.holder.load_champion()
        yield

    app = FastAPI(title="Recommendation API", version="1.0", lifespan=lifespan)

    def _model(request: Request) -> tuple[EmbeddingRecommender, str]:
        h: ModelHolder = request.app.state.holder
        if h.model is None:
            raise HTTPException(503, "Model not loaded")
        return h.model, h.version

    @app.get("/health")
    def health(request: Request) -> dict:
        h: ModelHolder = request.app.state.holder
        return {"status": "ok", "model_version": h.version, "feature_store": request.app.state.store.meta()}

    @app.get("/reco", response_model=RecoResponse)
    def recommend(request: Request, client_id: str, k: int = Query(10, ge=1, le=50)) -> RecoResponse:
        start = time.perf_counter()
        model, version = _model(request)
        features, realtime = request.app.state.store.get_customer(client_id)
        history = merge_history(realtime, features.recent_products, model.params.history_size)
        rec = model.recommend(history, features.segment, k)
        latency = (time.perf_counter() - start) * 1000
        logger.info("reco client=%s strategy=%s latency_ms=%.1f", client_id, rec.strategy, latency)
        return RecoResponse(
            client_id=client_id,
            product_ids=rec.product_ids,
            scores=rec.scores,
            strategy=rec.strategy,
            model_version=version,
            latency_ms=round(latency, 2),
        )

    @app.get("/reco/category/{category}", response_model=RecoResponse)
    def recommend_category(request: Request, category: str, k: int = Query(10, ge=1, le=50)) -> RecoResponse:
        start = time.perf_counter()
        model, version = _model(request)
        rec = model.recommend_category(category, k)
        if not rec.product_ids:
            raise HTTPException(404, f"Unknown category {category}")
        return RecoResponse(
            product_ids=rec.product_ids,
            scores=rec.scores,
            strategy=rec.strategy,
            model_version=version,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    @app.get("/reco/similar/{product_id}", response_model=RecoResponse)
    def similar(request: Request, product_id: str, k: int = Query(10, ge=1, le=50)) -> RecoResponse:
        start = time.perf_counter()
        model, version = _model(request)
        rec = model.similar_items(product_id, k)
        if not rec.product_ids:
            raise HTTPException(404, f"Unknown product {product_id}")
        return RecoResponse(
            product_ids=rec.product_ids,
            scores=rec.scores,
            strategy=rec.strategy,
            model_version=version,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    @app.post("/events", status_code=202)
    def ingest_event(request: Request, event: Event) -> dict:
        """Stand-in for the Kafka consumer that keeps real-time features fresh."""
        request.app.state.store.push_event(event.client_id, event.product_id)
        return {"accepted": True}

    @app.post("/admin/reload")
    def reload_model(request: Request) -> dict:
        """Hot-swap to the current champion after a promotion (or roll back by moving the alias)."""
        request.app.state.holder.load_champion()
        return {"model_version": request.app.state.holder.version}

    return app


app = create_app()
