import pandas as pd


def test_materialize_and_read(store):
    df = pd.DataFrame(
        [
            {
                "customer_id": "C1",
                "segment": "gardener",
                "favorite_category": "garden",
                "n_events_30d": 4,
                "recent_products": ["P1", "P2"],
                "feature_ts": "2026-11-01",
            }
        ]
    )
    assert store.materialize_customers(df) == 1
    features, recent = store.get_customer("C1")
    assert features.segment == "gardener" and features.recent_products == ["P1", "P2"]
    assert recent == []
    assert store.meta()["n_customers"] == "1"


def test_realtime_events_are_capped_and_recent_first(store):
    for i in range(8):
        store.push_event("C2", f"P{i}")
    _, recent = store.get_customer("C2")
    assert recent == ["P7", "P6", "P5", "P4", "P3"]  # recent_max=5


def test_unknown_customer_returns_empty_features(store):
    features, recent = store.get_customer("nobody")
    assert features.segment is None and features.recent_products == [] and recent == []
