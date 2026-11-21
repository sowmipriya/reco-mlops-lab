"""Offline comparison of baselines and model variants on the same evaluation set.

uv run python notebooks/compare_models.py
Reads the dbt-built tables once, then evaluates every variant on identical users and targets.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import pandas as pd

from reco.config import get_settings
from reco.data.bigquery import read_table
from reco.evaluation.metrics import evaluate
from reco.models.recommender import (
    EmbeddingRecommender,
    PopularityRecommender,
    RecentlyViewedRecommender,
    RecommenderParams,
)
from reco.pipelines.steps import _eval_inputs

warnings.filterwarnings("ignore")
K = 10
s = get_settings()
interactions = read_table(s.features_dataset, "ml_training_interactions")
products = read_table(s.marts_dataset, "dim_products", ["product_id", "category", "is_active"])
customers = read_table(s.marts_dataset, "dim_customers", ["customer_id", "segment"])
targets = read_table(s.features_dataset, "ml_eval_targets")
relevant, history, users, segments = _eval_inputs(interactions, customers, targets)
n_items = int(products["is_active"].sum())
histories = [history[u] for u in users]
seen = {u: set(h) for u, h in zip(users, histories, strict=True)}

rows = []


def record(name: str, recs: dict) -> None:
    m = evaluate(recs, relevant, K, n_items)
    hits = [p for u in users for p in recs[u][:K] if p in relevant[u]]
    m["share_of_hits_already_seen"] = sum(
        p in seen[u] for u, recs_u in recs.items() for p in recs_u[:K] if p in relevant[u]
    ) / max(len(hits), 1)
    rows.append({"model": name, **m})


pop = PopularityRecommender().fit(interactions)
record("global popularity", {u: pop.recommend(None, K) for u in users})
segpop = PopularityRecommender().fit(interactions, customers)
record("segment popularity (baseline)", {u: segpop.recommend(sg, K) for u, sg in zip(users, segments, strict=True)})
rv = RecentlyViewedRecommender().fit(interactions, customers)
record(
    "recently viewed + segment fill",
    {u: rv.recommend(h, sg, K) for u, h, sg in zip(users, histories, segments, strict=True)},
)

base = RecommenderParams()
variants = {
    "embeddings + segment prior (v4 champion)": base,
    "  ... 256 dims": replace(base, n_components=256),
    **{f"  ... 256 dims + seen boost {b}": replace(base, n_components=256, seen_boost=b) for b in (0.5, 1.0, 2.0, 5.0)},
    "  ... 128 dims + seen boost 2.0": replace(base, seen_boost=2.0),
    "  ... 256 dims + seen boost 2.0, decay 0.9": replace(base, n_components=256, seen_boost=2.0, recency_decay=0.9),
    "  ... 256 dims + seen boost 2.0, blend 0.5": replace(base, n_components=256, seen_boost=2.0, popularity_blend=0.5),
}
for name, params in variants.items():
    model = EmbeddingRecommender(params).fit(interactions, products, customers)
    recs = dict(zip(users, model.recommend_batch(histories, K, segments), strict=True))
    record(name, recs)

df = pd.DataFrame(rows).set_index("model")
cols = [
    f"ndcg_at_{K}",
    f"recall_at_{K}",
    f"map_at_{K}",
    f"hit_rate_at_{K}",
    f"coverage_at_{K}",
    "share_of_hits_already_seen",
]
pd.set_option("display.width", 200)
print(f"{len(users):,} evaluation customers, {n_items:,} active products\n")
print(df[cols].round(3).to_string())
