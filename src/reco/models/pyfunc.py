"""MLflow pyfunc packaging of the recommender, so any MLflow-aware serving stack can load it."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from reco.models.recommender import EmbeddingRecommender

ARTIFACT_KEY = "recommender"


class RecommenderPyfunc(mlflow.pyfunc.PythonModel):
    """Input: DataFrame with `history` (list of product ids, most recent first) and `segment`.
    Output: DataFrame with `product_ids` and `strategy`."""

    def load_context(self, context) -> None:
        with open(context.artifacts[ARTIFACT_KEY], "rb") as f:
            self.recommender: EmbeddingRecommender = pickle.load(f)

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        k = int((params or {}).get("k", 10))
        out = []
        for row in model_input.itertuples():
            rec = self.recommender.recommend(list(row.history), getattr(row, "segment", None), k)
            out.append({"product_ids": rec.product_ids, "strategy": rec.strategy})
        return pd.DataFrame(out)


def log_recommender(recommender: EmbeddingRecommender, registered_model_name: str) -> mlflow.models.model.ModelInfo:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "recommender.pkl"
        with open(path, "wb") as f:
            pickle.dump(recommender, f)
        example = pd.DataFrame(
            {"history": [list(recommender.item_ids[:3])], "segment": [next(iter(recommender.segment_top), None)]}
        )
        wrapper = RecommenderPyfunc()
        wrapper.recommender = recommender
        signature = infer_signature(example, wrapper.predict(None, example), params={"k": 10})
        return mlflow.pyfunc.log_model(
            name="model",
            python_model=wrapper,
            artifacts={ARTIFACT_KEY: str(path)},
            signature=signature,
            input_example=example,
            registered_model_name=registered_model_name,
            pip_requirements=["numpy", "pandas", "scipy", "scikit-learn"],
        )


def load_recommender(model_uri: str) -> EmbeddingRecommender:
    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    return pyfunc_model.unwrap_python_model().recommender
