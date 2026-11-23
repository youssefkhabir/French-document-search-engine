"""The two functions everything calls. app.py and eval/run_eval.py both use `search`,
so the system demonstrated is exactly the system measured.
"""
from pathlib import Path

MODES = ("tfidf", "bm25", "semantic", "hybrid", "word2vec")  # word2vec = optional S6


def build_index(folder: Path) -> dict:
    """extract -> clean -> chunk -> lexical forms -> TF-IDF, BM25, embeddings,
    vocabulary (+ optional word2vec) -> save to index/. Returns summary counts."""
    raise NotImplementedError("Phases 1-3")


def search(query: str, mode: str = "hybrid", alpha: float = 0.5, rerank: bool = False,
           typo_fix: bool = True, wordnet: bool = False, top_k: int = 10) -> dict:
    """correct -> branch search -> fusion -> rerank -> group by document.

    Returns {"query", "corrected_query", "corrections", "results": [
        {"doc_id", "title", "page", "score", "snippet", "rank_bm25", "rank_semantic"}],
        "timings": {...}}
    """
    raise NotImplementedError("Phases 2-4")
