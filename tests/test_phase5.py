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


# --- sampling and significance -----------------------------------------------------------

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eval import significance  # noqa: E402
from eval.run_eval import sample_queries  # noqa: E402


def test_sample_is_seeded_judged_and_ordered():
    queries = [{"qid": f"q{i}"} for i in range(50)]
    qrels = {f"q{i}": {"d": 1 if i % 5 else 0} for i in range(50)}   # every 5th unjudged
    s1, s2 = sample_queries(queries, qrels, 10), sample_queries(queries, qrels, 10)
    assert s1 == s2 and len(s1) == 10
    assert all(int(q["qid"][1:]) % 5 for q in s1)
    assert [q["qid"] for q in s1] == sorted((q["qid"] for q in s1), key=lambda x: int(x[1:]))
    assert len(sample_queries(queries, qrels, 1000)) == 40


def test_paired_bootstrap():
    a = np.array([0.9, 0.8, 0.7, 0.9, 0.6] * 20)
    p, lo, hi = significance.paired_bootstrap(a, a - 0.2)
    assert p < 0.01 and lo == pytest.approx(0.2) and hi == pytest.approx(0.2)
    p, lo, hi = significance.paired_bootstrap(a, a)
    assert p == 1.0


def test_compare_table():
    rows = [{"qid": q, "system": s, "ndcg@10": v}
            for q in range(30) for s, v in (("S3", 0.5), ("S2", 0.5 - 0.01 * (q % 3)))]
    t = significance.compare(pd.DataFrame(rows), pairs=[("S3", "S2"), ("S9", "S2")])
    assert list(t["comparison"]) == ["S3 vs S2"]
    assert t["wins"].iat[0] == 20 and t["ties"].iat[0] == 10
