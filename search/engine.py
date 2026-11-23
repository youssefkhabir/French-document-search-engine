"""The two functions everything calls. app.py and eval/run_eval.py both use `search`,
so the system demonstrated is exactly the system measured.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

import numpy as np

from search import config, lexical
from search.chunk import chunk_pages
from search.clean import clean_text, remove_repeated_lines
from search.extract import extract_folder

MODES = ("tfidf", "bm25", "semantic", "hybrid", "word2vec")  # word2vec = optional S6
MODES_READY = ("tfidf", "bm25")                               # grows with each phase
SNIPPET_WORDS = 45
CHUNK_COLUMNS = ["chunk_id", "doc_id", "collection", "format", "title", "page", "n_words",
                 "text_raw", "text_index", "meta"]


def ingest(folder: Path) -> tuple[pd.DataFrame, list[tuple[str, str]], dict]:
    """Phase 1: extract -> remove headers/footers -> clean -> chunk.

    Returns (chunks table, skipped files, report). The report keeps the removed
    headers/footers per document for the before/after example in the report.
    """
    pages, skipped = extract_folder(folder)

    by_doc: dict[str, list[dict]] = {}
    for p in pages:
        by_doc.setdefault(p["doc_id"], []).append(p)

    cleaned, removed = [], {}
    for doc_id, doc_pages in by_doc.items():
        doc_pages.sort(key=lambda p: p["page"])
        paginated = doc_pages[0]["format"] == "pdf"
        texts, dropped = remove_repeated_lines([p["text"] for p in doc_pages], paginated)
        if dropped:
            removed[doc_id] = dropped
        for p, text in zip(doc_pages, texts):
            text = clean_text(text)
            if text:
                cleaned.append({**p, "text": text})

    chunks = pd.DataFrame(chunk_pages(cleaned), columns=CHUNK_COLUMNS)
    # Parquet cannot store a column of (possibly empty) dicts: keep meta as JSON text.
    chunks["meta"] = chunks["meta"].map(lambda m: json.dumps(m or {}, ensure_ascii=False))
    report = {
        "n_documents": chunks["doc_id"].nunique(),
        "n_pages": len(pages),
        "n_chunks": len(chunks),
        "n_skipped": len(skipped),
        "per_collection": {c or "(root)": int(n) for c, n in
                           chunks.groupby("collection")["doc_id"].nunique().items()},
        "removed_headers_footers": removed,
    }
    return chunks, skipped, report


def build_index(folder: Path | None = None, index_dir: Path | None = None) -> dict:
    """extract -> clean -> chunk -> lexical forms -> TF-IDF, BM25, embeddings,
    vocabulary (+ optional word2vec) -> save to index/. Returns summary counts.

    Phase 1 implements the first step and writes chunks.parquet, skipped.json, meta.json.
    """
    folder = Path(folder or config.CORPUS_DIR)
    index_dir = Path(index_dir or config.INDEX_DIR)
    index_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    chunks, skipped, report = ingest(folder)
    if chunks.empty:
        raise ValueError(f"no indexable document in {folder}")
    (index_dir / "skipped.json").write_text(
        json.dumps([{"path": p, "reason": r} for p, r in skipped], ensure_ascii=False, indent=2),
        encoding="utf-8")

    # Phase 2 · lexical forms, TF-IDF, BM25.
    t_lex = time.perf_counter()
    chunks["text_lex"] = lexical.lexical_prep_many(chunks["text_index"].tolist())
    chunks.to_parquet(index_dir / "chunks.parquet", index=False)
    vectorizer, matrix = lexical.build_tfidf(chunks["text_lex"].tolist())
    lexical.save_tfidf(vectorizer, matrix, index_dir / "tfidf.pkl")
    lexical.save_bm25(lexical.build_bm25(chunks["text_lex"].tolist()), index_dir / "bm25")
    lexical_seconds = round(time.perf_counter() - t_lex, 2)

    # Phases 3-4 add: embeddings, vocabulary, word2vec.

    meta = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "corpus": str(folder),
        "n_documents": report["n_documents"],
        "n_pages": report["n_pages"],
        "n_chunks": report["n_chunks"],
        "n_skipped": report["n_skipped"],
        "documents_per_collection": report["per_collection"],
        "chunk_words": {"max": config.CHUNK_MAX_WORDS, "overlap": config.CHUNK_OVERLAP_WORDS},
        "title_prefix": config.CHUNK_TITLE_PREFIX,
        "lexical": {"vocabulary": len(vectorizer.vocabulary_), "seconds": lexical_seconds},
        "seconds": round(time.perf_counter() - t0, 2),
    }
    _INDEX_CACHE.pop(str(index_dir.resolve()), None)
    (index_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return {**meta, "skipped": skipped, "removed_headers_footers": report["removed_headers_footers"]}


_INDEX_CACHE: dict[str, dict] = {}


def load_index(index_dir: Path | None = None) -> dict:
    """Load chunks and the lexical indexes once per process (reloaded if rebuilt).

    Returns {"chunks", "meta", "tfidf": (vectorizer, matrix), "bm25"}.
    """
    index_dir = Path(index_dir or config.INDEX_DIR)
    key = str(index_dir.resolve())
    meta_path = index_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no index in {index_dir}: run scripts/build_index.py")
    stamp = meta_path.stat().st_mtime
    cached = _INDEX_CACHE.get(key)
    if cached and cached["stamp"] == stamp:
        return cached

    chunks = pd.read_parquet(index_dir / "chunks.parquet")
    if "text_lex" not in chunks.columns:
        raise FileNotFoundError(f"index in {index_dir} predates Phase 2: rebuild it")
    chunks["meta"] = chunks["meta"].map(json.loads)
    loaded = {
        "stamp": stamp,
        "dir": index_dir,
        "chunks": chunks,
        "meta": json.loads(meta_path.read_text(encoding="utf-8")),
        "tfidf": lexical.load_tfidf(index_dir / "tfidf.pkl"),
        "bm25": lexical.load_bm25(index_dir / "bm25"),
    }
    _INDEX_CACHE[key] = loaded
    return loaded


def _filter_mask(chunks: pd.DataFrame, collections, formats) -> np.ndarray | None:
    """Rows allowed by the collection / file type filters (None = no filter)."""
    mask = np.ones(len(chunks), dtype=bool)
    if collections:
        mask &= chunks["collection"].isin(list(collections)).to_numpy()
    if formats:
        mask &= chunks["format"].isin(list(formats)).to_numpy()
    return None if mask.all() else mask


def _group_by_document(hits: list[tuple[int, float]], chunks: pd.DataFrame,
                       top_k: int) -> list[tuple[int, float]]:
    """Chunk hits -> documents. A document's score is its best chunk's score (max-pooling);
    the best chunk gives the snippet and the page."""
    best: dict[str, tuple[int, float]] = {}
    doc_ids = chunks["doc_id"].to_numpy()
    for row, score in hits:  # hits are sorted best first
        best.setdefault(doc_ids[row], (row, score))
    return list(best.values())[:top_k]


def search(query: str, mode: str = "hybrid", alpha: float = 0.5, rerank: bool = False,
           typo_fix: bool = True, wordnet: bool = False, top_k: int = 10,
           collections: list[str] | None = None, formats: list[str] | None = None,
           index_dir: Path | None = None) -> dict:
    """correct -> branch search -> fusion -> rerank -> group by document.

    Phase 2 implements the lexical modes ("tfidf", "bm25"); typo correction, fusion,
    reranking and WordNet come in Phases 3-4 and are ignored until then.

    Returns {"query", "query_lex", "corrected_query", "corrections", "mode", "n_chunks_matched",
        "results": [{"doc_id", "title", "collection", "format", "page", "score", "snippet",
                     "highlights", "chunk_id", "meta", "rank_bm25", "rank_semantic"}],
        "timings": {...}}   (timings in milliseconds)
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
    if mode not in MODES_READY:
        raise NotImplementedError(f"mode {mode!r} arrives in a later phase")

    timings = {}
    t = time.perf_counter()
    idx = load_index(index_dir)
    chunks = idx["chunks"]
    timings["load"] = round((time.perf_counter() - t) * 1000, 1)

    t = time.perf_counter()
    query_lex = lexical.lexical_prep(query)
    timings["prep"] = round((time.perf_counter() - t) * 1000, 1)

    t = time.perf_counter()
    mask = _filter_mask(chunks, collections, formats)
    # Lexical scoring is cheap: rank every matching chunk, so a long document with many
    # matching chunks cannot push the other documents out of the top k.
    n_all = len(chunks)
    if mode == "tfidf":
        vectorizer, matrix = idx["tfidf"]
        hits = lexical.search_tfidf(query_lex, vectorizer, matrix, n_all, mask)
    else:
        hits = lexical.search_bm25(query_lex, idx["bm25"], n_all, mask)
    timings["search"] = round((time.perf_counter() - t) * 1000, 1)

    t = time.perf_counter()
    terms = set(query_lex.split())
    results = []
    for rank, (row, score) in enumerate(_group_by_document(hits, chunks, top_k), 1):
        c = chunks.iloc[row]
        text, spans = lexical.snippet(c["text_raw"], terms, SNIPPET_WORDS)
        results.append({
            "doc_id": c["doc_id"], "title": c["title"], "collection": c["collection"],
            "format": c["format"], "page": int(c["page"]), "score": round(score, 4),
            "snippet": text, "highlights": spans, "chunk_id": c["chunk_id"], "meta": c["meta"],
            "rank_bm25": rank if mode == "bm25" else None, "rank_semantic": None,
        })
    timings["snippets"] = round((time.perf_counter() - t) * 1000, 1)

    return {"query": query, "query_lex": query_lex, "corrected_query": query,
            "corrections": [], "mode": mode, "n_chunks_matched": len(hits),
            "results": results, "timings": timings}
