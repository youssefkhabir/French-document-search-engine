"""OPTIONAL (course Ch. 3) · Averaged word2vec baseline (system S6).

Default: gensim Word2Vec (skip-gram) trained on the lexical form of the chunks, the same
terms BM25 sees (lemmas, lowercase, no accents, no stopwords). Takes seconds, fully offline.
Optional: pretrained French KeyedVectors from config.W2V_PRETRAINED_PATH. Their keys are
raw words with accents, so they only match the part of our lexical forms that are
unchanged (lemma = word, no accent): a known limit of this option.
See docs/plan_addendum.md, component B.

A chunk vector is the mean of its word vectors (static: "souris" has one vector whatever
the context), optionally IDF-weighted; the query the same way; cosine; top-k.
Files: index/w2v.kv (word vectors) and index/w2v_matrix.npy (one row per chunk).
"""
import json
import time
from pathlib import Path

import numpy as np

from search import config


def train_or_load(texts_lex: list[str]):
    """KeyedVectors: pretrained if configured, else trained on the corpus lexical forms."""
    if config.W2V_PRETRAINED_PATH:
        from gensim.models import KeyedVectors
        path = Path(config.W2V_PRETRAINED_PATH)
        return KeyedVectors.load_word2vec_format(str(path), binary=path.suffix == ".bin")
    from gensim.models import KeyedVectors, Word2Vec
    sentences = [t.split() for t in texts_lex]
    counts: dict[str, int] = {}
    for s in sentences:
        for w in s:
            counts[w] = counts.get(w, 0) + 1
    if not any(c >= config.W2V_MIN_COUNT for c in counts.values()):
        return KeyedVectors(config.W2V_DIM)  # tiny corpus: no vocabulary, matches nothing
    model = Word2Vec(sentences=sentences, vector_size=config.W2V_DIM,
                     window=config.W2V_WINDOW, min_count=config.W2V_MIN_COUNT, sg=1,
                     epochs=config.W2V_EPOCHS, workers=config.W2V_WORKERS, seed=config.W2V_SEED)
    return model.wv


def idf_weights(texts_lex: list[str]) -> dict[str, float]:
    """Smoothed IDF of every term (same formula as scikit-learn's TfidfVectorizer)."""
    df: dict[str, int] = {}
    for t in texts_lex:
        for w in set(t.split()):
            df[w] = df.get(w, 0) + 1
    n = len(texts_lex)
    return {w: float(np.log((1 + n) / (1 + d)) + 1) for w, d in df.items()}


def doc_vector(kv, text_lex: str, idf: dict[str, float] | None = None) -> np.ndarray:
    """Mean of the word vectors (IDF-weighted if `idf`), L2-normalized; zero vector if no
    word is known."""
    words = [w for w in text_lex.split() if w in kv.key_to_index]
    if not words:
        return np.zeros(kv.vector_size, dtype=np.float32)
    weights = np.array([idf.get(w, 1.0) if idf else 1.0 for w in words], dtype=np.float32)
    v = (kv[words] * weights[:, None]).sum(axis=0) / weights.sum()
    norm = np.linalg.norm(v)
    return (v / norm).astype(np.float32) if norm > 0 else v.astype(np.float32)


def doc_matrix(kv, texts_lex: list[str], idf: dict[str, float] | None = None) -> np.ndarray:
    return np.vstack([doc_vector(kv, t, idf) for t in texts_lex]) if texts_lex else \
        np.zeros((0, kv.vector_size), dtype=np.float32)


def search_w2v(q_vec: np.ndarray, W: np.ndarray, top_k: int,
               mask: np.ndarray | None = None) -> list[tuple[int, float]]:
    """Cosine of the query with every chunk -> [(row, score)] best first (score > 0 only:
    a query with no known word, or an empty chunk, matches nothing)."""
    if not q_vec.any():
        return []
    scores = W @ q_vec
    if mask is not None:
        scores = np.where(mask, scores, 0.0)
    k = min(top_k, int((scores > 0).sum()))
    if k <= 0:
        return []
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(int(i), float(scores[i])) for i in idx]


def neighbours(kv, word: str, n: int = 8) -> list[tuple[str, float]]:
    """Nearest words by cosine (report: the distributional hypothesis). [] if unknown."""
    return kv.most_similar(word, topn=n) if word in kv.key_to_index else []


# --- index files ------------------------------------------------------------------

def build(texts_lex: list[str]) -> tuple[object, np.ndarray, dict]:
    """Train (or load) the vectors and embed every chunk -> (kv, W, info)."""
    t0 = time.perf_counter()
    kv = train_or_load(texts_lex)
    idf = idf_weights(texts_lex) if config.W2V_IDF_WEIGHTED else None
    W = doc_matrix(kv, texts_lex, idf)
    info = {"source": str(config.W2V_PRETRAINED_PATH or "trained on corpus"),
            "dim": kv.vector_size, "vocabulary": len(kv.key_to_index),
            "idf_weighted": config.W2V_IDF_WEIGHTED, "n": int(W.shape[0]),
            "seconds": round(time.perf_counter() - t0, 1)}
    return kv, W, info


def save(kv, W: np.ndarray, info: dict, index_dir: Path) -> None:
    kv.save(str(index_dir / "w2v.kv"))
    np.save(index_dir / "w2v_matrix.npy", W)
    (index_dir / "w2v.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def load(index_dir: Path):
    """(kv, W, info), or None when the index has no word2vec files."""
    if not (index_dir / "w2v_matrix.npy").exists():
        return None
    from gensim.models import KeyedVectors
    info = json.loads((index_dir / "w2v.json").read_text(encoding="utf-8"))
    return KeyedVectors.load(str(index_dir / "w2v.kv")), np.load(index_dir / "w2v_matrix.npy"), info
