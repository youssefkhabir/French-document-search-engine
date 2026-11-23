"""The two functions everything calls. app.py and eval/run_eval.py both use `search`,
so the system demonstrated is exactly the system measured.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from search import config
from search.chunk import chunk_pages
from search.clean import clean_text, remove_repeated_lines
from search.extract import extract_folder

MODES = ("tfidf", "bm25", "semantic", "hybrid", "word2vec")  # word2vec = optional S6
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
    chunks.to_parquet(index_dir / "chunks.parquet", index=False)
    (index_dir / "skipped.json").write_text(
        json.dumps([{"path": p, "reason": r} for p, r in skipped], ensure_ascii=False, indent=2),
        encoding="utf-8")

    # Phases 2-4 add: lexical forms, TF-IDF, BM25, embeddings, vocabulary, word2vec.

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
        "seconds": round(time.perf_counter() - t0, 2),
    }
    (index_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return {**meta, "skipped": skipped, "removed_headers_footers": report["removed_headers_footers"]}


def search(query: str, mode: str = "hybrid", alpha: float = 0.5, rerank: bool = False,
           typo_fix: bool = True, wordnet: bool = False, top_k: int = 10) -> dict:
    """correct -> branch search -> fusion -> rerank -> group by document.

    Returns {"query", "corrected_query", "corrections", "results": [
        {"doc_id", "title", "page", "score", "snippet", "rank_bm25", "rank_semantic"}],
        "timings": {...}}
    """
    raise NotImplementedError("Phases 2-4")
