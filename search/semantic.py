"""Phase 3 · Semantic branch: transformer embeddings + cosine similarity.

Vectors are L2-normalized, so `scores = E @ q` is the cosine similarity.
e5 models need "query: " / "passage: " prefixes (see config.EMBED_PREFIXES); BGE-M3 needs none.

The index records which model built it (embeddings.json); queries are always embedded with
that model, whatever config.EMBED_MODEL says, so index and query vectors stay comparable.
"""
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from search import config


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def load_model(name: str | None = None):
    """Load a SentenceTransformer once per process: GPU in fp16 if available, else CPU."""
    from sentence_transformers import SentenceTransformer

    name = name or config.EMBED_MODEL
    dev = device()
    model = SentenceTransformer(name, device=dev)
    if dev == "cuda":
        model.half()  # fp16: about 2x faster and half the VRAM, same ranking in practice
    # Chunks are ~250 words (~400 tokens). The cap keeps memory bounded on 6 GB GPUs;
    # BGE-M3 would otherwise accept 8,192 tokens.
    model.max_seq_length = min(model.max_seq_length, config.EMBED_MAX_TOKENS)
    return model


def _prefixes(name: str) -> tuple[str, str]:
    return config.EMBED_PREFIXES.get(name, ("", ""))


def embed_passages(model, texts: list[str], name: str | None = None,
                   batch_size: int | None = None, show_progress: bool = False) -> np.ndarray:
    """Embed corpus chunks -> float32 array (n, dim), rows L2-normalized."""
    p_pre = _prefixes(name or config.EMBED_MODEL)[1]
    E = model.encode([p_pre + t for t in texts], batch_size=batch_size or config.EMBED_BATCH,
                     normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=show_progress)
    return E.astype(np.float32)


def embed_query(model, query: str, name: str | None = None) -> np.ndarray:
    q_pre = _prefixes(name or config.EMBED_MODEL)[0]
    v = model.encode([q_pre + query], normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=False)[0]
    return v.astype(np.float32)


def search_semantic(q_vec: np.ndarray, E: np.ndarray, top_k: int,
                    mask: np.ndarray | None = None) -> list[tuple[int, float]]:
    """Cosine similarity of the query with every chunk -> [(row, score)] best first."""
    scores = E @ q_vec
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    k = min(top_k, int(np.isfinite(scores).sum()))
    if k <= 0:
        return []
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(int(i), float(scores[i])) for i in idx]


# --- index files ------------------------------------------------------------------

def build_embeddings(texts: list[str], name: str | None = None, show_progress: bool = False) -> tuple[np.ndarray, dict]:
    """Embed all chunks; returns (E, info) where info goes to embeddings.json."""
    name = name or config.EMBED_MODEL
    t0 = time.perf_counter()
    model = load_model(name)
    load_s = time.perf_counter() - t0
    n_long = sum(len(ids) > model.max_seq_length
                 for ids in model.tokenizer([_prefixes(name)[1] + t for t in texts])["input_ids"])
    t1 = time.perf_counter()
    E = embed_passages(model, texts, name, show_progress=show_progress)
    info = {"model": name, "dim": int(E.shape[1]), "n": int(E.shape[0]), "device": device(),
            "max_tokens": model.max_seq_length, "n_truncated": int(n_long),
            "load_seconds": round(load_s, 1), "seconds": round(time.perf_counter() - t1, 1)}
    return E, info


def save_embeddings(E: np.ndarray, info: dict, index_dir: Path) -> None:
    np.save(index_dir / "embeddings.npy", E)
    (index_dir / "embeddings.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def load_embeddings(index_dir: Path) -> tuple[np.ndarray, dict] | None:
    """(E, info), or None when the index was built without embeddings."""
    path = index_dir / "embeddings.npy"
    if not path.exists():
        return None
    info = json.loads((index_dir / "embeddings.json").read_text(encoding="utf-8"))
    return np.load(path), info
