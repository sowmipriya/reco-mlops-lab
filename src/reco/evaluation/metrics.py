"""Offline Top-K ranking metrics for recommendation.

All functions take:
    recommendations: {user_id: [item ids ranked best-first]}
    relevant:        {user_id: {ground-truth item ids}}
and average over users that have at least one relevant item.
"""

from __future__ import annotations

import numpy as np


def _users(recommendations: dict, relevant: dict) -> list:
    return [u for u in recommendations if relevant.get(u)]


def precision_at_k(recommendations: dict, relevant: dict, k: int) -> float:
    users = _users(recommendations, relevant)
    return float(np.mean([len(set(recommendations[u][:k]) & relevant[u]) / k for u in users])) if users else 0.0


def recall_at_k(recommendations: dict, relevant: dict, k: int) -> float:
    users = _users(recommendations, relevant)
    return (
        float(np.mean([len(set(recommendations[u][:k]) & relevant[u]) / len(relevant[u]) for u in users]))
        if users
        else 0.0
    )


def hit_rate_at_k(recommendations: dict, relevant: dict, k: int) -> float:
    """Share of users with at least one relevant item in the top k."""
    users = _users(recommendations, relevant)
    return float(np.mean([bool(set(recommendations[u][:k]) & relevant[u]) for u in users])) if users else 0.0


def ndcg_at_k(recommendations: dict, relevant: dict, k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance: rewards hits near the top."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    scores = []
    for u in _users(recommendations, relevant):
        gains = np.array([1.0 if item in relevant[u] else 0.0 for item in recommendations[u][:k]])
        dcg = float((gains * discounts[: len(gains)]).sum())
        idcg = float(discounts[: min(len(relevant[u]), k)].sum())
        scores.append(dcg / idcg)
    return float(np.mean(scores)) if scores else 0.0


def map_at_k(recommendations: dict, relevant: dict, k: int) -> float:
    """Mean average precision: average of precision at each hit position."""
    scores = []
    for u in _users(recommendations, relevant):
        hits, total = 0, 0.0
        for rank, item in enumerate(recommendations[u][:k], start=1):
            if item in relevant[u]:
                hits += 1
                total += hits / rank
        scores.append(total / min(len(relevant[u]), k))
    return float(np.mean(scores)) if scores else 0.0


def catalog_coverage(recommendations: dict, catalog_size: int, k: int) -> float:
    """Share of the catalogue that appears in at least one top-k list (diversity / long tail)."""
    shown = {item for recs in recommendations.values() for item in recs[:k]}
    return len(shown) / catalog_size if catalog_size else 0.0


def evaluate(recommendations: dict, relevant: dict, k: int, catalog_size: int) -> dict[str, float]:
    return {
        f"precision_at_{k}": precision_at_k(recommendations, relevant, k),
        f"recall_at_{k}": recall_at_k(recommendations, relevant, k),
        f"ndcg_at_{k}": ndcg_at_k(recommendations, relevant, k),
        f"map_at_{k}": map_at_k(recommendations, relevant, k),
        f"hit_rate_at_{k}": hit_rate_at_k(recommendations, relevant, k),
        f"coverage_at_{k}": catalog_coverage(recommendations, catalog_size, k),
        "n_eval_users": float(len(_users(recommendations, relevant))),
    }
