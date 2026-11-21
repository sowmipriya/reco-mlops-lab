import numpy as np


def test_embeddings_are_normalised(model):
    norms = np.linalg.norm(model.item_embeddings, axis=1)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)


def test_personalised_recommendation(model, interactions):
    customer = interactions["customer_id"].iloc[0]
    history = interactions[interactions["customer_id"] == customer].sort_values("last_ts", ascending=False)
    rec = model.recommend(history["product_id"].tolist(), "diy_expert", k=5)
    assert rec.strategy == "personalized"
    assert len(rec.product_ids) == 5 == len(set(rec.product_ids))
    assert rec.scores == sorted(rec.scores, reverse=True)


def test_cold_start_falls_back_to_segment_then_global(model):
    assert model.recommend([], "gardener", k=3).strategy == "segment_popularity"
    assert model.recommend(["UNKNOWN"], None, k=3).strategy == "global_popularity"


def test_category_and_similar(model, tables):
    rec = model.recommend_category("garden", k=4)
    cats = tables["products"].set_index("product_id").loc[rec.product_ids, "category"]
    assert (cats == "garden").all()
    pid = model.item_ids[0]
    sim = model.similar_items(pid, k=3)
    assert pid not in sim.product_ids and len(sim.product_ids) == 3


def test_recommendations_follow_history_category(model, tables):
    """A customer who only browsed paint should mostly get paint back."""
    products = tables["products"].set_index("product_id")
    paint = products[products["category"] == "paint"].index[:5].tolist()
    rec = model.recommend(paint, None, k=10)
    share = (products.loc[rec.product_ids, "category"] == "paint").mean()
    assert share >= 0.5


def test_batch_matches_single(model, interactions):
    histories = interactions.groupby("customer_id")["product_id"].apply(list).head(5).tolist()
    batch = model.recommend_batch(histories, k=5)
    single = [model.recommend(h, None, 5).product_ids for h in histories]
    assert batch == single
