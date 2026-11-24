"""Phase 5 tests: evaluation metrics."""
import math

import pytest

from eval import metrics

QRELS = {"a": 2, "b": 1, "c": 0}


def test_recall():
    assert metrics.recall_at_k(["x", "a", "y"], QRELS, 10) == 0.5
    assert metrics.recall_at_k(["x", "b", "a"], QRELS, 10) == 1.0
    assert metrics.recall_at_k(["x", "y", "a"], QRELS, 2) == 0.0


def test_mrr():
    assert metrics.mrr_at_k(["c", "x", "b"], QRELS, 10) == pytest.approx(1 / 3)
    assert metrics.mrr_at_k(["c", "x"], QRELS, 10) == 0.0


def test_ndcg():
    assert metrics.ndcg_at_k(["a", "b"], QRELS, 10) == pytest.approx(1.0)
    ideal = 3 + 1 / math.log2(3)
    assert metrics.ndcg_at_k(["b", "a"], QRELS, 10) == pytest.approx((1 + 3 / math.log2(3)) / ideal)
    assert metrics.ndcg_at_k(["x"], {"x": 0}, 10) == 0.0


def test_has_relevant():
    assert metrics.has_relevant(QRELS)
    assert not metrics.has_relevant({"c": 0})
