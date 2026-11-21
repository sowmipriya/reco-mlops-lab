"""ZenML materializer: store the recommender as numpy arrays + JSON instead of pickle,
so artifacts are portable across Python versions and inspectable."""

from __future__ import annotations

import io
import json
import os
from dataclasses import asdict
from typing import Any, ClassVar

import numpy as np
from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

from reco.models.recommender import EmbeddingRecommender, RecommenderParams


class EmbeddingRecommenderMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES: ClassVar[tuple[type[Any], ...]] = (EmbeddingRecommender,)
    ASSOCIATED_ARTIFACT_TYPE: ClassVar[ArtifactType] = ArtifactType.MODEL

    def save(self, data: EmbeddingRecommender) -> None:
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            item_ids=data.item_ids.astype(str),
            item_embeddings=data.item_embeddings,
            item_popularity=data.item_popularity,
            item_category=data.item_category.astype(str),
            **{f"segpop__{s}": v for s, v in data.segment_popularity.items()},
        )
        with fileio.open(os.path.join(self.uri, "arrays.npz"), "wb") as f:
            f.write(buffer.getvalue())
        meta = {
            "params": asdict(data.params),
            "segment_top": data.segment_top,
            "explained_variance": data.explained_variance,
        }
        with fileio.open(os.path.join(self.uri, "meta.json"), "w") as f:
            f.write(json.dumps(meta))

    def load(self, data_type: type[Any]) -> EmbeddingRecommender:
        with fileio.open(os.path.join(self.uri, "meta.json"), "r") as f:
            meta = json.loads(f.read())
        with fileio.open(os.path.join(self.uri, "arrays.npz"), "rb") as f:
            arrays = np.load(io.BytesIO(f.read()), allow_pickle=False)
            model = EmbeddingRecommender(params=RecommenderParams(**meta["params"]))
            model.item_ids = arrays["item_ids"].astype(object)
            model.item_embeddings = arrays["item_embeddings"]
            model.item_popularity = arrays["item_popularity"]
            model.item_category = arrays["item_category"].astype(object)
            model.segment_popularity = {
                k.removeprefix("segpop__"): arrays[k] for k in arrays.files if k.startswith("segpop__")
            }
        model.segment_top = meta["segment_top"]
        model.explained_variance = meta["explained_variance"]
        return model

    def extract_metadata(self, data: EmbeddingRecommender) -> dict[str, Any]:
        return {
            "n_items": len(data.item_ids),
            "embedding_dim": int(data.item_embeddings.shape[1]),
            "explained_variance": data.explained_variance,
        }
