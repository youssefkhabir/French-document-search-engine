"""Phase 5 · Recall@k, MRR@k, nDCG@k written by hand (formulas stay visible for the report).

qrels: {doc_id: grade} with grade in {0, 1, 2}; ranking: list of doc_id, best first.
A document is relevant when its grade is >= 1. Queries with no relevant document are
excluded from averages and reported separately (the caller checks `has_relevant`).
"""
import math


def has_relevant(qrels: dict[str, int]) -> bool:
    return any(g > 0 for g in qrels.values())


def recall_at_k(ranking: list[str], qrels: dict[str, int], k: int = 10) -> float:
    """Share of the relevant documents found in the top k."""
    relevant = {d for d, g in qrels.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranking[:k])) / len(relevant)


def mrr_at_k(ranking: list[str], qrels: dict[str, int], k: int = 10) -> float:
    """1 / rank of the first relevant document in the top k, 0 if none."""
    for rank, doc in enumerate(ranking[:k], start=1):
        if qrels.get(doc, 0) > 0:
            return 1.0 / rank
    return 0.0


def _dcg(grades: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(rank + 1) for rank, g in enumerate(grades, start=1))


def ndcg_at_k(ranking: list[str], qrels: dict[str, int], k: int = 10) -> float:
    """DCG = sum((2**grade - 1) / log2(rank + 1)); nDCG = DCG / ideal DCG."""
    ideal = _dcg(sorted((g for g in qrels.values() if g > 0), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return _dcg([qrels.get(d, 0) for d in ranking[:k]]) / ideal
