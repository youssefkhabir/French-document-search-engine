"""Phase 3 · Semantic branch: transformer embeddings + cosine similarity.

Vectors are L2-normalized, so `scores = E @ q` is the cosine similarity.
e5 models need "query: " / "passage: " prefixes (see config.EMBED_PREFIXES); BGE-M3 needs none.
"""
import numpy as np


def load_model(name: str | None = None):
    """Load a SentenceTransformer on GPU if available, else CPU."""
    raise NotImplementedError("Phase 3")


def embed_passages(model, texts: list[str]) -> np.ndarray:
    raise NotImplementedError("Phase 3")


def embed_query(model, query: str) -> np.ndarray:
    raise NotImplementedError("Phase 3")


def search_semantic(q_vec: np.ndarray, E: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    raise NotImplementedError("Phase 3")
