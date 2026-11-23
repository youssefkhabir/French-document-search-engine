"""Phase 4 · Cross-encoder reranking (system S5).

A bi-encoder (Phase 3) embeds the query and the chunk separately; a cross-encoder reads
them together, so it sees which words answer which. Far more precise, far too slow for the
whole corpus: it only reorders the RERANK_TOP_K best candidates of another mode.
"""
from functools import lru_cache

from search import config


@lru_cache(maxsize=1)
def load_reranker(name: str | None = None):
    """Load the cross-encoder once per process: GPU in fp16 if available, else CPU."""
    from sentence_transformers import CrossEncoder

    from search.semantic import device

    dev = device()
    model = CrossEncoder(name or config.RERANK_MODEL, device=dev,
                         max_length=config.RERANK_MAX_TOKENS)
    if dev == "cuda":
        model.model.half()
    return model


def rerank(model, query: str, candidates: list[tuple[int, str]]) -> list[tuple[int, float]]:
    """candidates: (chunk_index, text). Returns (chunk_index, score) sorted desc.

    Scores are the model's raw logits: comparable within one query only.
    """
    if not candidates:
        return []
    scores = model.predict([(query, text) for _, text in candidates],
                           batch_size=config.RERANK_BATCH, show_progress_bar=False)
    ranked = sorted(zip((row for row, _ in candidates), scores), key=lambda x: -float(x[1]))
    return [(row, float(s)) for row, s in ranked]
