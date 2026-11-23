"""Phase 4 (optional) · Cross-encoder reranking of the top RERANK_TOP_K fused chunks."""


def load_reranker(name: str | None = None):
    raise NotImplementedError("Phase 4")


def rerank(model, query: str, candidates: list[tuple[int, str]]) -> list[tuple[int, float]]:
    """candidates: (chunk_index, text_raw). Returns (chunk_index, score) sorted desc."""
    raise NotImplementedError("Phase 4")
