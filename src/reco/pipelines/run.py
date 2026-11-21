"""ZenML pipeline definitions and CLI entrypoint.

uv run python -m reco.pipelines.run train
uv run python -m reco.pipelines.run materialize
"""

from __future__ import annotations

import argparse

from zenml import pipeline

from reco.config import get_settings
from reco.pipelines.steps import (
    default_params,
    evaluate_baseline,
    evaluate_model,
    load_customer_features,
    load_training_data,
    materialize_online,
    register_and_promote,
    train_model,
    validate_training_data,
)


@pipeline(enable_cache=False)
def training_pipeline(params: dict, k: int = 10, min_uplift: float = 0.02) -> None:
    interactions, products, customers, eval_targets = load_training_data()
    ok = validate_training_data(interactions, eval_targets)
    model = train_model(interactions, products, customers, params, ok)
    model_metrics = evaluate_model(model, interactions, customers, eval_targets, k)
    baseline_metrics = evaluate_baseline(interactions, customers, eval_targets, products, k)
    register_and_promote(model, model_metrics, baseline_metrics, params, k, min_uplift)


@pipeline(enable_cache=False)
def feature_materialization_pipeline() -> None:
    materialize_online(load_customer_features())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", choices=["train", "materialize"])
    parser.add_argument("--n-components", type=int)
    parser.add_argument("--popularity-blend", type=float)
    parser.add_argument("--min-uplift", type=float, default=0.02)
    args = parser.parse_args()

    if args.pipeline == "train":
        params = default_params()
        if args.n_components:
            params["n_components"] = args.n_components
        if args.popularity_blend is not None:
            params["popularity_blend"] = args.popularity_blend
        training_pipeline(params=params, k=get_settings().top_k, min_uplift=args.min_uplift)
    else:
        feature_materialization_pipeline()


if __name__ == "__main__":
    main()
