"""ZenML steps for the recommendation training and feature materialisation pipelines."""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from zenml import log_metadata, step

from reco.config import get_settings
from reco.data.bigquery import read_table
from reco.evaluation.metrics import evaluate
from reco.feature_store.online import OnlineFeatureStore
from reco.models.pyfunc import log_recommender
from reco.models.recommender import EmbeddingRecommender, RecentlyViewedRecommender, RecommenderParams
from reco.pipelines.materializers import EmbeddingRecommenderMaterializer


# ------------------------------------------------------------------ data
@step
def load_training_data() -> tuple[
    Annotated[pd.DataFrame, "interactions"],
    Annotated[pd.DataFrame, "products"],
    Annotated[pd.DataFrame, "customers"],
    Annotated[pd.DataFrame, "eval_targets"],
]:
    """Read the dbt-built, point-in-time correct training and evaluation tables."""
    s = get_settings()
    interactions = read_table(s.features_dataset, "ml_training_interactions")
    products = read_table(s.marts_dataset, "dim_products", ["product_id", "category", "is_active"])
    customers = read_table(s.marts_dataset, "dim_customers", ["customer_id", "segment"])
    targets = read_table(s.features_dataset, "ml_eval_targets")
    log_metadata(
        metadata={
            "n_interactions": len(interactions),
            "n_customers": int(interactions["customer_id"].nunique()),
            "n_products": len(products),
            "n_eval_rows": len(targets),
        }
    )
    return interactions, products, customers, targets


@step
def validate_training_data(interactions: pd.DataFrame, eval_targets: pd.DataFrame) -> bool:
    """Fail fast if upstream data looks broken (dbt tests run before, this is a last guard)."""
    if len(interactions) < 10_000:
        raise ValueError(f"Too few training interactions: {len(interactions)}")
    if interactions["weight"].le(0).any():
        raise ValueError("Non-positive interaction weights")
    if eval_targets.empty:
        raise ValueError("Empty evaluation set")
    return True


# ------------------------------------------------------------- training
@step(output_materializers=EmbeddingRecommenderMaterializer)
def train_model(
    interactions: pd.DataFrame,
    products: pd.DataFrame,
    customers: pd.DataFrame,
    params: dict,
    data_ok: bool,
) -> Annotated[EmbeddingRecommender, "recommender"]:
    model = EmbeddingRecommender(RecommenderParams(**params)).fit(interactions, products, customers)
    log_metadata(metadata={"explained_variance": model.explained_variance})
    return model


def _eval_inputs(interactions: pd.DataFrame, customers: pd.DataFrame, eval_targets: pd.DataFrame):
    relevant = eval_targets.groupby("customer_id")["product_id"].apply(set).to_dict()
    history = interactions.sort_values("last_ts", ascending=False).groupby("customer_id")["product_id"].apply(list)
    users = [u for u in history.index if u in relevant]
    segments = customers.set_index("customer_id")["segment"].reindex(users).tolist()
    return relevant, history, users, segments


@step
def evaluate_model(
    model: EmbeddingRecommender,
    interactions: pd.DataFrame,
    customers: pd.DataFrame,
    eval_targets: pd.DataFrame,
    k: int,
) -> Annotated[dict, "model_metrics"]:
    relevant, history, users, segments = _eval_inputs(interactions, customers, eval_targets)
    recs = dict(zip(users, model.recommend_batch([history[u] for u in users], k, segments), strict=True))
    metrics = evaluate(recs, relevant, k, catalog_size=len(model.item_ids))
    log_metadata(metadata={"model": metrics})
    return metrics


@step
def evaluate_baseline(
    interactions: pd.DataFrame,
    customers: pd.DataFrame,
    eval_targets: pd.DataFrame,
    products: pd.DataFrame,
    k: int,
) -> Annotated[dict, "baseline_metrics"]:
    """Recently viewed + segment popularity fill: the bar any model must clear.
    On real data this heuristic beats plain popularity by 4x, so it is the honest baseline."""
    relevant, history, users, segments = _eval_inputs(interactions, customers, eval_targets)
    baseline = RecentlyViewedRecommender().fit(interactions, customers)
    recs = {u: baseline.recommend(history[u], s, k) for u, s in zip(users, segments, strict=True)}
    metrics = evaluate(recs, relevant, k, catalog_size=int(products["is_active"].sum()))
    log_metadata(metadata={"baseline": metrics})
    return metrics


# --------------------------------------------------- registry / promotion
@step
def register_and_promote(
    model: EmbeddingRecommender,
    model_metrics: dict,
    baseline_metrics: dict,
    params: dict,
    k: int,
    min_uplift: float = 0.02,
) -> Annotated[str, "registered_version"]:
    """Log to MLflow, register a new version, and move the `champion` alias only if the
    candidate beats both the baseline (by `min_uplift` NDCG) and the current champion."""
    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment(s.mlflow_experiment)
    client = MlflowClient()
    key = f"ndcg_at_{k}"

    with mlflow.start_run(run_name="embedding-recommender") as run:
        mlflow.log_params(params | {"k": k})
        mlflow.log_metrics(model_metrics)
        mlflow.log_metrics({f"baseline_{m}": v for m, v in baseline_metrics.items()})
        mlflow.log_metric("explained_variance", model.explained_variance)
        info = log_recommender(model, s.model_name)
        mlflow.set_tag("training_pipeline", "zenml")
        mlflow.set_tag("data_source", s.data_source)

    version = str(info.registered_model_version)
    candidate = model_metrics[key]
    beats_baseline = candidate >= baseline_metrics[key] * (1 + min_uplift)

    try:
        champion = client.get_model_version_by_alias(s.model_name, s.champion_alias)
        champion_score = client.get_run(champion.run_id).data.metrics.get(key, 0.0)
    except MlflowException:
        champion, champion_score = None, 0.0

    # Strictly better than the current champion: avoids alias churn between equivalent models.
    promoted = beats_baseline and (champion is None or candidate > champion_score)
    client.set_model_version_tag(s.model_name, version, "validation_status", "passed" if promoted else "rejected")
    client.set_model_version_tag(s.model_name, version, key, f"{candidate:.4f}")
    client.set_model_version_tag(s.model_name, version, "data_source", s.data_source)
    if promoted:
        client.set_registered_model_alias(s.model_name, s.champion_alias, version)

    log_metadata(
        metadata={
            "mlflow_run_id": run.info.run_id,
            "model_version": version,
            "promoted": promoted,
            "candidate_ndcg": candidate,
            "baseline_ndcg": baseline_metrics[key],
            "previous_champion_ndcg": champion_score,
        }
    )
    print(
        f"Model version {version}: ndcg={candidate:.4f} baseline={baseline_metrics[key]:.4f} "
        f"champion={champion_score:.4f} -> {'PROMOTED' if promoted else 'not promoted'}"
    )
    return version


# ------------------------------------------------- feature materialisation
@step
def load_customer_features() -> Annotated[pd.DataFrame, "customer_features"]:
    s = get_settings()
    df = read_table(
        s.features_dataset,
        "feat_customer_profile",
        ["customer_id", "segment", "favorite_category", "n_events_30d", "recent_products", "feature_ts"],
    )
    log_metadata(metadata={"n_customers": len(df)})
    return df


@step
def materialize_online(customer_features: pd.DataFrame) -> Annotated[int, "n_materialized"]:
    s = get_settings()
    store = OnlineFeatureStore.from_url(s.redis_url, s.recent_events_max)
    n = store.materialize_customers(customer_features)
    log_metadata(metadata={"n_materialized": n})
    return n


def default_params() -> dict:
    return asdict(RecommenderParams())
