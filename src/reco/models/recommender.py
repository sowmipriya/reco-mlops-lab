"""Embedding-based recommender with popularity fallbacks.

Three recommendation levels, as in production retail recommenders:
1. personalised  - user vector built from the customer's recent history (incl. real-time events)
2. segment       - most popular products for the customer's segment (cold start)
3. category      - most popular products in a category (category pages, no user context)

Item embeddings come from a truncated SVD of the (customer x product) implicit-feedback
matrix: items bought/viewed by the same customers end up close in embedding space.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


@dataclass
class RecommenderParams:
    n_components: int = 128
    recency_decay: float = 0.95  # weight multiplier per position in the recency-ordered history
    popularity_blend: float = 1.0  # weight of the popularity prior vs embedding similarity
    segment_prior: bool = True  # use the customer's segment popularity as prior when known
    exclude_seen: bool = False  # re-engaging with a seen product is common (comparison shopping)
    seen_boost: float = 2.0  # extra score for products in the history, scaled by recency (re-engagement)
    history_size: int = 20
    seed: int = 42


@dataclass
class Recommendation:
    product_ids: list[str]
    scores: list[float]
    strategy: str


@dataclass
class EmbeddingRecommender:
    params: RecommenderParams = field(default_factory=RecommenderParams)

    # Fitted state
    item_ids: np.ndarray | None = None
    item_embeddings: np.ndarray | None = None  # (n_items, n_components), L2-normalised
    item_popularity: np.ndarray | None = None  # (n_items,), scaled to [0, 1]
    item_category: np.ndarray | None = None
    segment_top: dict[str, list[str]] = field(default_factory=dict)
    segment_popularity: dict[str, np.ndarray] = field(default_factory=dict)
    explained_variance: float = 0.0

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        interactions: pd.DataFrame,
        products: pd.DataFrame,
        customers: pd.DataFrame | None = None,
    ) -> EmbeddingRecommender:
        """interactions: customer_id, product_id, weight. products: product_id, category[, is_active]."""
        active = products[products.get("is_active", True) == True]  # noqa: E712
        self.item_ids = active["product_id"].to_numpy()
        self._item_index = None  # rebuilt lazily for the new catalogue
        self.item_category = active["category"].to_numpy()
        item_index = {pid: i for i, pid in enumerate(self.item_ids)}

        data = interactions[interactions["product_id"].isin(item_index)]
        users = data["customer_id"].astype("category")
        rows = users.cat.codes.to_numpy()
        cols = data["product_id"].map(item_index).to_numpy()
        values = np.log1p(data["weight"].to_numpy(dtype=np.float64))
        matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(users.cat.categories), len(self.item_ids)))

        n_components = min(self.params.n_components, min(matrix.shape) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=self.params.seed)
        svd.fit(matrix)
        self.explained_variance = float(svd.explained_variance_ratio_.sum())
        # Item factors V * sqrt(S), normalised so dot product = cosine similarity.
        factors = svd.components_.T * np.sqrt(svd.singular_values_)
        self.item_embeddings = normalize(factors).astype(np.float32)

        popularity = np.asarray(matrix.sum(axis=0)).ravel()
        self.item_popularity = (np.log1p(popularity) / np.log1p(popularity.max())).astype(np.float32)

        if customers is not None:
            seg = data.merge(customers[["customer_id", "segment"]], on="customer_id")
            top = (
                seg.groupby(["segment", "product_id"])["weight"]
                .sum()
                .reset_index()
                .sort_values(["segment", "weight"], ascending=[True, False])
            )
            self.segment_top = {s: g["product_id"].head(100).tolist() for s, g in top.groupby("segment")}
            for s, g in top.groupby("segment"):
                vec = np.zeros(len(self.item_ids), dtype=np.float64)
                vec[g["product_id"].map(item_index).to_numpy()] = g["weight"].to_numpy()
                self.segment_popularity[s] = (np.log1p(vec) / np.log1p(vec.max())).astype(np.float32)
        return self

    # ------------------------------------------------------------ inference
    @property
    def item_index(self) -> dict[str, int]:
        """product_id -> row, built once (it is used on every request)."""
        if getattr(self, "_item_index", None) is None:
            self._item_index = {pid: i for i, pid in enumerate(self.item_ids)}
        return self._item_index

    def _history_matrix(self, histories: list[list[str]]) -> sparse.csr_matrix:
        """Sparse (n_users, n_items) matrix of recency-decayed history weights."""
        index = self.item_index
        rows, cols, vals = [], [], []
        for r, history in enumerate(histories):
            seen: set[int] = set()
            for pos, pid in enumerate(history[: self.params.history_size]):
                i = index.get(pid)
                if i is None or i in seen:
                    continue
                seen.add(i)
                rows.append(r)
                cols.append(i)
                vals.append(self.params.recency_decay**pos)
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(histories), len(self.item_ids)))

    def _prior(self, segments: list[str | None] | None, n: int) -> np.ndarray:
        if not (self.params.segment_prior and segments):
            return np.broadcast_to(self.item_popularity, (n, len(self.item_ids)))
        return np.stack([self.segment_popularity.get(s, self.item_popularity) for s in segments])

    def score_batch(self, histories: list[list[str]], segments: list[str | None] | None = None) -> np.ndarray:
        """Scores for every item, for a batch of recency-ordered histories (most recent first)."""
        hist = self._history_matrix(histories)
        user_vectors = normalize(hist @ self.item_embeddings)
        scores = user_vectors @ self.item_embeddings.T
        scores += self.params.popularity_blend * self._prior(segments, len(histories))
        if self.params.seen_boost:
            coo = hist.tocoo()
            scores[coo.row, coo.col] += self.params.seen_boost * coo.data
        if self.params.exclude_seen:
            scores[hist.nonzero()] = -np.inf
        return scores

    def recommend_batch(
        self,
        histories: list[list[str]],
        k: int = 10,
        segments: list[str | None] | None = None,
        batch_size: int = 1024,
    ) -> list[list[str]]:
        """Top-k for many users, scored in chunks so memory stays at batch_size x n_items."""
        out: list[list[str]] = []
        for start in range(0, len(histories), batch_size):
            chunk_segments = segments[start : start + batch_size] if segments else None
            scores = self.score_batch(histories[start : start + batch_size], chunk_segments)
            top = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
            ordered = np.take_along_axis(top, np.argsort(-np.take_along_axis(scores, top, 1), 1), 1)
            out.extend(self.item_ids[row].tolist() for row in ordered)
        return out

    def recommend(self, history: list[str], segment: str | None = None, k: int = 10) -> Recommendation:
        """Personalised if the customer has known history, else segment / global popularity."""
        known = [p for p in history if p in self.item_index]
        if known:
            scores = self.score_batch([known], [segment])[0]
            top = np.argsort(-scores)[:k]
            return Recommendation(
                self.item_ids[top].tolist(), scores[top].astype(float).round(4).tolist(), "personalized"
            )
        if segment and segment in self.segment_top:
            return Recommendation(self.segment_top[segment][:k], [], "segment_popularity")
        top = np.argsort(-self.item_popularity)[:k]
        return Recommendation(
            self.item_ids[top].tolist(), self.item_popularity[top].astype(float).round(4).tolist(), "global_popularity"
        )

    def recommend_category(self, category: str, k: int = 10) -> Recommendation:
        mask = self.item_category == category
        if not mask.any():
            return Recommendation([], [], "unknown_category")
        idx = np.flatnonzero(mask)
        top = idx[np.argsort(-self.item_popularity[idx])[:k]]
        return Recommendation(
            self.item_ids[top].tolist(),
            self.item_popularity[top].astype(float).round(4).tolist(),
            "category_popularity",
        )

    def similar_items(self, product_id: str, k: int = 10) -> Recommendation:
        """'Customers who bought this also bought' via embedding similarity."""
        matches = np.flatnonzero(self.item_ids == product_id)
        if len(matches) == 0:
            return Recommendation([], [], "unknown_product")
        sims = self.item_embeddings @ self.item_embeddings[matches[0]]
        sims[matches[0]] = -np.inf
        top = np.argsort(-sims)[:k]
        return Recommendation(self.item_ids[top].tolist(), sims[top].astype(float).round(4).tolist(), "item_similarity")


class RecentlyViewedRecommender:
    """Strong heuristic baseline on real e-commerce data: show the customer's most recent
    products again, then fill the list with their segment's most popular products."""

    def __init__(self) -> None:
        self.popularity = PopularityRecommender()

    def fit(self, interactions: pd.DataFrame, customers: pd.DataFrame | None = None) -> RecentlyViewedRecommender:
        self.popularity.fit(interactions, customers)
        return self

    def recommend(self, history: list[str], segment: str | None = None, k: int = 10) -> list[str]:
        out = list(dict.fromkeys(history))[:k]
        for pid in self.popularity.recommend(segment, k + len(out)):
            if len(out) == k:
                break
            if pid not in out:
                out.append(pid)
        return out


class PopularityRecommender:
    """Baseline: same top-k popular items for everyone (segment-aware if available)."""

    def __init__(self) -> None:
        self.top: list[str] = []
        self.segment_top: dict[str, list[str]] = {}

    def fit(self, interactions: pd.DataFrame, customers: pd.DataFrame | None = None) -> PopularityRecommender:
        self.top = interactions.groupby("product_id")["weight"].sum().sort_values(ascending=False).index.tolist()
        if customers is not None:
            seg = interactions.merge(customers[["customer_id", "segment"]], on="customer_id")
            for s, g in seg.groupby("segment"):
                self.segment_top[s] = (
                    g.groupby("product_id")["weight"].sum().sort_values(ascending=False).index.tolist()
                )
        return self

    def recommend(self, segment: str | None = None, k: int = 10) -> list[str]:
        return self.segment_top.get(segment, self.top)[:k]
