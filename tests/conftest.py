import fakeredis
import pandas as pd
import pytest

from reco.data.generate import GeneratorConfig, generate
from reco.feature_store.online import OnlineFeatureStore
from reco.models.recommender import EmbeddingRecommender, RecommenderParams


@pytest.fixture(scope="session")
def tables() -> dict[str, pd.DataFrame]:
    return generate(GeneratorConfig(n_customers=400, n_products_per_subcategory=10, seed=7))


@pytest.fixture(scope="session")
def interactions(tables) -> pd.DataFrame:
    weights = {"view": 1.0, "add_to_cart": 3.0, "purchase": 5.0}
    ev = tables["events"].assign(weight=lambda d: d["event_type"].map(weights))
    return (
        ev.groupby(["customer_id", "product_id"])
        .agg(weight=("weight", "sum"), last_ts=("event_ts", "max"))
        .reset_index()
    )


@pytest.fixture(scope="session")
def model(tables, interactions) -> EmbeddingRecommender:
    return EmbeddingRecommender(RecommenderParams(n_components=16)).fit(
        interactions, tables["products"], tables["customers"]
    )


@pytest.fixture
def store() -> OnlineFeatureStore:
    return OnlineFeatureStore(fakeredis.FakeRedis(decode_responses=True), recent_max=5)
