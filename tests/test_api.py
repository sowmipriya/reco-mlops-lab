import pytest
from fastapi.testclient import TestClient

from reco.serving.app import ModelHolder, create_app, merge_history


@pytest.fixture
def client(model, store):
    holder = ModelHolder()
    holder.model, holder.version = model, "test"
    with TestClient(create_app(store=store, holder=holder)) as c:
        yield c


def test_merge_history_dedupes_and_keeps_realtime_first():
    assert merge_history(["b", "a"], ["a", "c"], limit=3) == ["b", "a", "c"]


def test_health(client):
    assert client.get("/health").json()["model_version"] == "test"


def test_cold_start_then_realtime_personalisation(client, model):
    first = client.get("/reco", params={"client_id": "new", "k": 5}).json()
    assert first["strategy"] == "global_popularity"
    for pid in model.item_ids[:3]:
        assert client.post("/events", json={"client_id": "new", "product_id": pid}).status_code == 202
    second = client.get("/reco", params={"client_id": "new", "k": 5}).json()
    assert second["strategy"] == "personalized"
    assert len(second["product_ids"]) == 5


def test_category_and_validation(client):
    assert client.get("/reco/category/paint", params={"k": 3}).json()["strategy"] == "category_popularity"
    assert client.get("/reco/category/unknown").status_code == 404
    assert client.get("/reco", params={"client_id": "x", "k": 0}).status_code == 422


def test_features_shows_batch_and_realtime_history(client):
    client.post("/events", json={"client_id": "NEW_F", "product_id": "P_RT"})
    body = client.get("/features/NEW_F").json()
    assert body["known_customer"] is False
    assert body["realtime_history"] == ["P_RT"] and body["merged_history"][0] == "P_RT"
