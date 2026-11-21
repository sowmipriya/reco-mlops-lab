"""Online feature store backed by Redis.

Two kinds of features, as in the production design:
- pre-built (batch) features: computed by dbt in BigQuery, materialised here by a pipeline
- real-time features: the customer's latest events, pushed by a stream consumer (Kafka in
  production, the API's /events endpoint here) and read at request time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd
import redis

CUSTOMER_KEY = "fs:customer:{customer_id}"
RECENT_KEY = "fs:recent:{customer_id}"
META_KEY = "fs:meta"


@dataclass
class CustomerFeatures:
    customer_id: str
    segment: str | None = None
    favorite_category: str | None = None
    n_events_30d: int = 0
    recent_products: list[str] = field(default_factory=list)
    feature_ts: str | None = None


class OnlineFeatureStore:
    def __init__(self, client: redis.Redis, recent_max: int = 20) -> None:
        self.r = client
        self.recent_max = recent_max

    @classmethod
    def from_url(cls, url: str, recent_max: int = 20) -> OnlineFeatureStore:
        return cls(redis.Redis.from_url(url, decode_responses=True), recent_max)

    # ----------------------------------------------------------- batch path
    def materialize_customers(self, df: pd.DataFrame, batch_size: int = 1000) -> int:
        """Upsert pre-built customer features (one Redis hash per customer)."""
        written = 0
        for start in range(0, len(df), batch_size):
            pipe = self.r.pipeline(transaction=False)
            for row in df.iloc[start : start + batch_size].itertuples():
                recent = list(row.recent_products) if row.recent_products is not None else []
                pipe.hset(
                    CUSTOMER_KEY.format(customer_id=row.customer_id),
                    mapping={
                        "segment": row.segment or "",
                        "favorite_category": row.favorite_category or "",
                        "n_events_30d": int(row.n_events_30d),
                        "recent_products": json.dumps(recent),
                        "feature_ts": str(row.feature_ts),
                    },
                )
                written += 1
            pipe.execute()
        self.r.hset(
            META_KEY,
            mapping={"last_materialization": datetime.now(UTC).isoformat(), "n_customers": written},
        )
        return written

    # ------------------------------------------------------- real-time path
    def push_event(self, customer_id: str, product_id: str) -> None:
        key = RECENT_KEY.format(customer_id=customer_id)
        pipe = self.r.pipeline()
        pipe.lpush(key, product_id)
        pipe.ltrim(key, 0, self.recent_max - 1)
        pipe.expire(key, 7 * 24 * 3600)  # real-time signal only matters for a few days
        pipe.execute()

    # ------------------------------------------------------------- serving
    def get_customer(self, customer_id: str) -> tuple[CustomerFeatures, list[str]]:
        """Pre-built features and real-time recent products, in a single round trip."""
        pipe = self.r.pipeline(transaction=False)
        pipe.hgetall(CUSTOMER_KEY.format(customer_id=customer_id))
        pipe.lrange(RECENT_KEY.format(customer_id=customer_id), 0, self.recent_max - 1)
        stored, recent = pipe.execute()
        features = CustomerFeatures(
            customer_id=customer_id,
            segment=stored.get("segment") or None,
            favorite_category=stored.get("favorite_category") or None,
            n_events_30d=int(stored.get("n_events_30d", 0)),
            recent_products=json.loads(stored.get("recent_products", "[]")),
            feature_ts=stored.get("feature_ts"),
        )
        return features, recent

    def meta(self) -> dict:
        return self.r.hgetall(META_KEY)
