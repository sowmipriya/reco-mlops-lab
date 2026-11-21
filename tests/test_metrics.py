import pytest

from reco.evaluation import metrics as m

RECS = {"u1": ["a", "b", "c"], "u2": ["x", "y", "z"], "u3": ["a"]}
REL = {"u1": {"a", "c"}, "u2": {"q"}, "u3": set()}  # u3 has no ground truth -> ignored


def test_precision_recall_hit():
    assert m.precision_at_k(RECS, REL, 3) == pytest.approx((2 / 3 + 0) / 2)
    assert m.recall_at_k(RECS, REL, 3) == pytest.approx((1.0 + 0) / 2)
    assert m.hit_rate_at_k(RECS, REL, 3) == pytest.approx(0.5)


def test_ndcg_perfect_and_partial():
    assert m.ndcg_at_k({"u": ["a", "b"]}, {"u": {"a", "b"}}, 2) == pytest.approx(1.0)
    # hit at rank 2 only: (1/log2(3)) / (1/log2(2))
    assert m.ndcg_at_k({"u": ["x", "a"]}, {"u": {"a"}}, 2) == pytest.approx(0.6309, abs=1e-4)


def test_map():
    # hits at rank 1 and 3 -> (1/1 + 2/3) / 2
    assert m.map_at_k({"u": ["a", "x", "c"]}, {"u": {"a", "c"}}, 3) == pytest.approx((1 + 2 / 3) / 2)


def test_coverage():
    assert m.catalog_coverage(RECS, catalog_size=10, k=3) == pytest.approx(0.6)


def test_empty_inputs_return_zero():
    assert m.ndcg_at_k({}, {}, 10) == 0.0
