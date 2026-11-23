"""Phase 5 · Recall@k, MRR@k, nDCG@k written by hand (formulas stay visible for the report).

qrels: {doc_id: grade} with grade in {0, 1, 2}; ranking: list of doc_id, best first.
Queries with no relevant document are excluded from averages and reported separately.
"""


def recall_at_k(ranking: list[str], qrels: dict[str, int], k: int = 10) -> float:
    raise NotImplementedError("Phase 5")


def mrr_at_k(ranking: list[str], qrels: dict[str, int], k: int = 10) -> float:
    raise NotImplementedError("Phase 5")


def ndcg_at_k(ranking: list[str], qrels: dict[str, int], k: int = 10) -> float:
    """DCG = sum((2**grade - 1) / log2(rank + 1)); nDCG = DCG / ideal DCG."""
    raise NotImplementedError("Phase 5")
