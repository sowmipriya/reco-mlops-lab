"""Demo storefront: the "website + website backend" boxes in front of the Reco API.

    Browser -> this backend (:8020) -> Reco API (:8010) /features, /reco, /reco/category, /reco/similar
                                    -> Kafka topic reco.events (clicks) -> consumer -> Redis real-time features

The backend owns the product catalogue (read once from the BigQuery catalogue mart) and turns
product ids returned by the Reco API into product cards. Every upstream call is returned in a
`trace` so the page can show what happened behind the scenes.

    uv run uvicorn reco.website.app:app --port 8020
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from reco.config import get_settings
from reco.streaming.events import ClickEvent
from reco.streaming.producer import EventProducer

RECO_API_URL = os.environ.get("RECO_API_URL", "http://127.0.0.1:8010")
STATIC = Path(__file__).parent / "static"
K = 12


class CartItem(BaseModel):
    client_id: str
    product_id: str


def load_catalogue() -> tuple[dict[str, dict], list[dict]]:
    """Products from the catalogue mart, plus one real sample customer per segment."""
    from reco.data.bigquery import client

    s = get_settings()
    bq = client()
    products = bq.query(
        f"select product_id, product_name, category, subcategory, brand, cast(price as float64) as price "
        f"from `{s.gcp_project}.{s.marts_dataset}.dim_products` where is_active"
    ).to_dataframe(create_bqstorage_client=False)
    samples = bq.query(
        f"""
        select segment,
               array_agg(customer_id order by n_events_30d desc, customer_id limit 1)[offset(0)] as customer_id,
               max(n_events_30d) as n_events_30d
        from `{s.gcp_project}.{s.features_dataset}.feat_customer_profile`
        where segment != 'new_customer' and n_events_30d between 15 and 40
        group by segment order by segment
        """
    ).to_dataframe(create_bqstorage_client=False)
    products["brand"] = products["brand"].fillna("")
    return {r["product_id"]: r for r in products.to_dict("records")}, samples.to_dict("records")


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.catalogue, app.state.samples = load_catalogue()
        app.state.http = httpx.Client(base_url=RECO_API_URL, timeout=5.0)
        s = get_settings()
        app.state.events = EventProducer(s.kafka_bootstrap, s.events_topic)
        yield
        app.state.http.close()

    app = FastAPI(title="DIY store demo", lifespan=lifespan)

    def call(request: Request, trace: list, method: str, path: str, **kwargs) -> dict:
        start = time.perf_counter()
        response = request.app.state.http.request(method, path, **kwargs)
        body = response.json() if response.content else {}
        trace.append(
            {
                "call": f"{method} {path}",
                "status": response.status_code,
                "round_trip_ms": round((time.perf_counter() - start) * 1000, 1),
                "api_latency_ms": body.get("latency_ms"),
                "strategy": body.get("strategy"),
                "model_version": body.get("model_version"),
            }
        )
        if response.status_code >= 400:
            raise HTTPException(response.status_code, body.get("detail", "Reco API error"))
        return body

    def publish(request: Request, trace: list, event: ClickEvent) -> dict:
        """Clicks go to Kafka, not to the Reco API: the website does not wait for the features update."""
        try:
            delivery = request.app.state.events.publish(event)
        except RuntimeError as e:
            raise HTTPException(503, f"{e}. Is Redpanda running (make kafka)?") from e
        trace.append(
            {
                "call": f"PRODUCE {delivery['topic']} ({event.event_type})",
                "status": f"partition {delivery['partition']}, offset {delivery['offset']}",
                "round_trip_ms": delivery["ms"],
                "api_latency_ms": None,
                "strategy": None,
                "model_version": None,
            }
        )
        return delivery

    def cards(request: Request, reco: dict) -> list[dict]:
        catalogue = request.app.state.catalogue
        scores = reco.get("scores") or [None] * len(reco["product_ids"])
        return [
            catalogue[pid] | {"score": score}
            for pid, score in zip(reco["product_ids"], scores, strict=False)
            if pid in catalogue
        ]

    def history_cards(request: Request, features: dict) -> list[dict]:
        catalogue = request.app.state.catalogue
        realtime = set(features["realtime_history"])
        return [
            catalogue[pid] | {"source": "real-time" if pid in realtime else "batch"}
            for pid in features["merged_history"][:8]
            if pid in catalogue
        ]

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/meta")
    def meta(request: Request) -> dict:
        departments = sorted({p["category"] for p in request.app.state.catalogue.values()})
        return {
            "departments": departments,
            "sample_customers": request.app.state.samples,
            "n_products": len(request.app.state.catalogue),
        }

    @app.get("/api/home")
    def home(request: Request, client_id: str) -> dict:
        trace: list = []
        features = call(request, trace, "GET", f"/features/{client_id}")
        reco = call(request, trace, "GET", "/reco", params={"client_id": client_id, "k": K})
        return {
            "features": features,
            "history": history_cards(request, features),
            "recommended": cards(request, reco),
            "strategy": reco["strategy"],
            "trace": trace,
        }

    @app.get("/api/department/{department}")
    def department(request: Request, department: str) -> dict:
        trace: list = []
        reco = call(request, trace, "GET", f"/reco/category/{department}", params={"k": K})
        return {"department": department, "products": cards(request, reco), "trace": trace}

    @app.get("/api/product/{product_id}")
    def product(request: Request, product_id: str, client_id: str) -> dict:
        if product_id not in request.app.state.catalogue:
            raise HTTPException(404, "Unknown product")
        trace: list = []
        # Opening a product page is a view event: it updates the customer's real-time features.
        publish(request, trace, ClickEvent(client_id=client_id, product_id=product_id, event_type="view"))
        similar = call(request, trace, "GET", f"/reco/similar/{product_id}", params={"k": 8})
        return {
            "product": request.app.state.catalogue[product_id],
            "similar": cards(request, similar),
            "trace": trace,
        }

    @app.post("/api/cart")
    def add_to_cart(request: Request, item: CartItem) -> dict:
        trace: list = []
        publish(
            request, trace, ClickEvent(client_id=item.client_id, product_id=item.product_id, event_type="add_to_cart")
        )
        return {"added": True, "trace": trace}

    return app


app = create_app()
