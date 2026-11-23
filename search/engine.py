"""The two functions everything calls. app.py and eval/run_eval.py both use `search`,
so the system demonstrated is exactly the system measured.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

import numpy as np

from search import config, fusion, lexical, semantic, spelling
from search import rerank as rerank_mod
from search.chunk import chunk_pages
from search.clean import clean_text, remove_repeated_lines
from search.extract import extract_folder

MODES = ("tfidf", "bm25", "semantic", "hybrid", "word2vec")  # word2vec = optional S6
MODES_READY = ("tfidf", "bm25", "semantic", "hybrid")         # grows with each phase
NEEDS_EMBEDDINGS = ("semantic", "hybrid")
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


def build_index(folder: Path | None = None, index_dir: Path | None = None, embed: bool = True,
                embed_model: str | None = None, show_progress: bool = False) -> dict:
    """extract -> clean -> chunk -> lexical forms -> TF-IDF, BM25, embeddings,
    vocabulary (+ optional word2vec) -> save to index/. Returns summary counts.

    `embed=False` skips the embeddings (quick lexical-only build); `embed_model` overrides
    config.EMBED_MODEL (the tests use the small e5 model).
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

    # Phase 3 · embeddings. A lexical-only build removes stale ones so they can't be
    # paired with chunks they don't belong to.
    embeddings_info = None
    if embed:
        E, embeddings_info = semantic.build_embeddings(chunks["text_index"].tolist(), embed_model,
                                                       show_progress)
        semantic.save_embeddings(E, embeddings_info, index_dir)
    else:
        for name in ("embeddings.npy", "embeddings.json"):
            (index_dir / name).unlink(missing_ok=True)

    # Phase 4 · vocabulary for typo correction (raw words, accents kept).
    vocab = spelling.build_vocabulary(chunks["text_raw"].tolist())
    spelling.save_vocabulary(vocab, index_dir / "vocab.json")

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
        "embeddings": embeddings_info,
        "spelling_vocabulary": len(vocab),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    _INDEX_CACHE.pop(str(index_dir.resolve()), None)
    (index_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return {**meta, "skipped": skipped, "removed_headers_footers": report["removed_headers_footers"]}


_INDEX_CACHE: dict[str, dict] = {}


def load_index(index_dir: Path | None = None) -> dict:
    """Load chunks and the lexical indexes once per process (reloaded if rebuilt).

    Returns {"chunks", "meta", "tfidf": (vectorizer, matrix), "bm25",
             "embeddings": (E, info) or None, "modes": modes this index can serve,
             "speller": spelling.Speller}.
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
        "embeddings": semantic.load_embeddings(index_dir),   # (E, info) or None
    }
    if loaded["embeddings"] is not None and len(loaded["embeddings"][0]) != len(chunks):
        raise ValueError(f"embeddings in {index_dir} don't match chunks.parquet: rebuild the index")
    loaded["modes"] = tuple(m for m in MODES_READY
                            if m not in NEEDS_EMBEDDINGS or loaded["embeddings"] is not None)
    # Typo correction vocabulary. Indexes built before Phase 4 have no vocab.json: it is
    # rebuilt from the chunks (about a second).
    vocab_path = index_dir / "vocab.json"
    vocab = (spelling.load_vocabulary(vocab_path) if vocab_path.exists()
             else spelling.build_vocabulary(chunks["text_raw"].tolist()))
    loaded["speller"] = spelling.Speller(vocab)
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


def _doc_ranks(hits: list[tuple[int, float]], chunks: pd.DataFrame) -> dict[str, int]:
    """Rank of each document in one branch's chunk list (its best chunk's position)."""
    ranks: dict[str, int] = {}
    doc_ids = chunks["doc_id"].to_numpy()
    for row, _ in hits:
        ranks.setdefault(doc_ids[row], len(ranks) + 1)
    return ranks


def _ms(t: float) -> float:
    return round((time.perf_counter() - t) * 1000, 1)


def search(query: str, mode: str = "hybrid", alpha: float | None = None, rerank: bool = False,
           typo_fix: bool = True, wordnet: bool = False, top_k: int = 10,
           collections: list[str] | None = None, formats: list[str] | None = None,
           index_dir: Path | None = None, fusion_method: str | None = None,
           snippets: bool = True) -> dict:
    """correct -> branch search -> fusion -> rerank -> group by document.

    mode: "tfidf", "bm25", "semantic" or "hybrid" (BM25 + semantic, fused).
    alpha: weight of the semantic branch in weighted fusion (default config.DEFAULT_ALPHA).
    fusion_method: "weighted" or "rrf" (default config.FUSION).
    rerank: reorder the best chunk of the top RERANK_TOP_K documents with the cross-encoder.
    typo_fix: correct the query against the corpus vocabulary first (spelling.py).
    snippets: False skips the snippets (the evaluation only needs the ranking).
    WordNet expansion is optional (Phase 4 stretch) and ignored for now.

    Single modes rank every chunk the filters allow (a long document cannot crowd out the
    others). Hybrid fuses the top BRANCH_TOP_K chunks of each branch, as the plan says:
    min-max over a branch's whole score range would squash the scores that matter.

    Returns {"query", "query_lex", "corrected_query", "corrections", "mode", "alpha",
        "fusion", "reranked", "n_chunks_matched",
        "results": [{"doc_id", "title", "collection", "format", "page", "score", "snippet",
                     "highlights", "chunk_id", "meta", "rank_bm25", "rank_semantic",
                     "rerank_score"}],
        "timings": {...}}   (timings in milliseconds)
    rank_bm25 / rank_semantic: the document's rank in that branch, when the mode ran it.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
    if mode not in MODES_READY:
        raise NotImplementedError(f"mode {mode!r} arrives in a later phase")
    alpha = config.DEFAULT_ALPHA if alpha is None else alpha
    fusion_method = fusion_method or config.FUSION
    if fusion_method not in ("weighted", "rrf"):
        raise ValueError(f"unknown fusion {fusion_method!r}")

    timings = {}
    t = time.perf_counter()
    idx = load_index(index_dir)
    chunks = idx["chunks"]
    timings["load"] = _ms(t)
    if mode in NEEDS_EMBEDDINGS and idx["embeddings"] is None:
        raise ValueError("this index was built without embeddings: rebuild it without --no-embed")

    t = time.perf_counter()
    corrected, corrections = (spelling.correct_query(query, idx["speller"]) if typo_fix
                              else (query, []))
    timings["correct"] = _ms(t)

    t = time.perf_counter()
    query_lex = lexical.lexical_prep(corrected)
    timings["prep"] = _ms(t)

    if mode in NEEDS_EMBEDDINGS:
        E, info = idx["embeddings"]
        t = time.perf_counter()
        # The transformer reads typos through its sub-word tokens; a "correction" can change
        # the meaning. Unless configured otherwise it gets the original query, only
        # lowercased when typed in capitals (which it does handle badly).
        sem_query = (corrected if config.SEMANTIC_USES_CORRECTION or not typo_fix
                     else spelling.fix_capitals(query))
        q_vec = semantic.embed_query(semantic.load_model(info["model"]), sem_query, info["model"])
        timings["embed"] = _ms(t)

    t = time.perf_counter()
    mask = _filter_mask(chunks, collections, formats)
    n_all = len(chunks)
    lex_hits = sem_hits = None
    if mode == "tfidf":
        vectorizer, matrix = idx["tfidf"]
        hits = lexical.search_tfidf(query_lex, vectorizer, matrix, n_all, mask)
    elif mode == "bm25":
        hits = lex_hits = lexical.search_bm25(query_lex, idx["bm25"], n_all, mask)
    elif mode == "semantic":
        hits = sem_hits = semantic.search_semantic(q_vec, E, n_all, mask)
    else:
        lex_hits = lexical.search_bm25(query_lex, idx["bm25"], config.BRANCH_TOP_K, mask)
        sem_hits = semantic.search_semantic(q_vec, E, config.BRANCH_TOP_K, mask)
    timings["search"] = _ms(t)

    if mode == "hybrid":
        t = time.perf_counter()
        hits = (fusion.weighted_fusion(lex_hits, sem_hits, alpha) if fusion_method == "weighted"
                else fusion.rrf_fusion(lex_hits, sem_hits, config.RRF_K))
        timings["fusion"] = _ms(t)

    n_docs = max(top_k, config.RERANK_TOP_K) if rerank else top_k
    docs = _group_by_document(hits, chunks, n_docs)
    rerank_scores: dict[int, float] = {}
    if rerank and docs:
        # Rerank the best chunk of each of the top RERANK_TOP_K documents. Reranking chunks
        # instead could leave fewer than top_k documents once grouped.
        t = time.perf_counter()
        head = docs[:config.RERANK_TOP_K]
        reranked = rerank_mod.rerank(rerank_mod.load_reranker(), corrected,
                                     [(row, chunks["text_index"].iat[row]) for row, _ in head])
        rerank_scores = dict(reranked)
        docs = [(row, s) for row, s in reranked] + docs[config.RERANK_TOP_K:]
        timings["rerank"] = _ms(t)
    docs = docs[:top_k]

    bm25_ranks = _doc_ranks(lex_hits, chunks) if lex_hits is not None else {}
    sem_ranks = _doc_ranks(sem_hits, chunks) if sem_hits is not None else {}
    t = time.perf_counter()
    terms = set(query_lex.split())
    results = []
    for row, score in docs:
        c = chunks.iloc[row]
        text, spans = (lexical.snippet(c["text_raw"], terms, SNIPPET_WORDS) if snippets
                       else ("", []))
        results.append({
            "doc_id": c["doc_id"], "title": c["title"], "collection": c["collection"],
            "format": c["format"], "page": int(c["page"]), "score": round(float(score), 4),
            "snippet": text, "highlights": spans, "chunk_id": c["chunk_id"], "meta": c["meta"],
            "rank_bm25": bm25_ranks.get(c["doc_id"]) if lex_hits is not None else None,
            "rank_semantic": sem_ranks.get(c["doc_id"]) if sem_hits is not None else None,
            "rerank_score": rerank_scores.get(row),
        })
    timings["snippets"] = _ms(t)

    return {"query": query, "query_lex": query_lex, "corrected_query": corrected,
            "corrections": corrections, "mode": mode,
            "alpha": alpha if mode == "hybrid" and fusion_method == "weighted" else None,
            "fusion": fusion_method if mode == "hybrid" else None, "reranked": bool(rerank),
            "n_chunks_matched": len(hits), "results": results, "timings": timings}
