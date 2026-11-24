"""Phase 5 · Ablations: what each preprocessing choice is worth.

Variants of the lexical branch (BM25 + typo, as S2) and of the semantic branch (S3), each
rebuilt in memory from the stored chunks (index/chunks.parquet), so the real index is not
touched. `text_raw` is the chunk without the title, `text_index` the chunk with it.

    lexical  baseline       lemmas, stopwords removed, title prefix   (= S2)
    lexical  raw words      no lemmatization
    lexical  stopwords kept
    lexical  no title       chunk text only
    semantic baseline       title prefix                               (= S3)
    semantic no title       chunk text only (re-embeds every chunk: ~4 min on GPU)

Each variant's chunk forms / embeddings are cached in eval/results/ablations/cache/.
Usage: python -m eval.ablations [--no-semantic]
Scores on the 300 dev questions and the 100-question test sample.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from search import config  # noqa: E402  (sets HF_HUB_CACHE before any HF import)
from search import engine, lexical, semantic, spelling  # noqa: E402
from eval import metrics  # noqa: E402
from eval.run_eval import K, load_queries, sample_queries  # noqa: E402

OUT = config.EVAL_DIR / "results" / "ablations"
CACHE = OUT / "cache"

LEXICAL_VARIANTS = {
    "baseline": {"field": "text_index", "lemmatize": True, "remove_stopwords": True},
    "raw words": {"field": "text_index", "lemmatize": False, "remove_stopwords": True},
    "stopwords kept": {"field": "text_index", "lemmatize": True, "remove_stopwords": False},
    "no title": {"field": "text_raw", "lemmatize": True, "remove_stopwords": True},
}
SEMANTIC_VARIANTS = {"baseline": "text_index", "no title": "text_raw"}


def _score(ranking_fn, queries, qrels, chunks) -> dict[str, float]:
    rows = []
    for q in queries:
        judged = qrels[q["qid"]]
        hits = ranking_fn(q["text"])
        ranking = [chunks["doc_id"].iat[r] for r, _ in engine._group_by_document(hits, chunks, K)]
        rows.append((metrics.recall_at_k(ranking, judged, K), metrics.mrr_at_k(ranking, judged, K),
                     metrics.ndcg_at_k(ranking, judged, K)))
    r, m, n = np.mean(rows, axis=0)
    return {"recall@10": round(r, 3), "mrr@10": round(m, 3), "ndcg@10": round(n, 3)}


def lexical_variant(name: str, opts: dict, chunks: pd.DataFrame) -> list[str]:
    path = CACHE / f"lex_{name.replace(' ', '_')}.parquet"
    if path.exists():
        return pd.read_parquet(path)["text_lex"].tolist()
    t = time.perf_counter()
    forms = lexical.lexical_prep_many(chunks[opts["field"]].tolist(), opts["lemmatize"],
                                      opts["remove_stopwords"])
    pd.DataFrame({"text_lex": forms}).to_parquet(path, index=False)
    print(f"  lexical forms '{name}' in {time.perf_counter() - t:.0f} s", flush=True)
    return forms


def semantic_variant(name: str, field: str, chunks: pd.DataFrame, idx: dict) -> np.ndarray:
    if field == "text_index":
        return idx["embeddings"][0]
    path = CACHE / f"emb_{name.replace(' ', '_')}.npy"
    if path.exists():
        return np.load(path)
    t = time.perf_counter()
    model_name = idx["embeddings"][1]["model"]
    E = semantic.embed_passages(semantic.load_model(model_name), chunks[field].tolist(),
                                model_name, show_progress=True)
    np.save(path, E)
    print(f"  embeddings '{name}' in {time.perf_counter() - t:.0f} s", flush=True)
    return E


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-semantic", action="store_true", help="skip the semantic ablation")
    args = p.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    idx = engine.load_index()
    chunks, speller = idx["chunks"], idx["speller"]
    queries, qrels, _ = load_queries("alloprof", None)
    sets = {"dev (300)": sample_queries([q for q in queries if q["split"] == "dev"], qrels, 10**9),
            "test (100)": sample_queries([q for q in queries if q["split"] == "test"], qrels, 100)}
    corrected = {q["qid"]: spelling.correct_query(q["text"], speller)[0]
                 for qs in sets.values() for q in qs}
    by_text = {q["text"]: q["qid"] for qs in sets.values() for q in qs}

    rows = []
    for name, opts in LEXICAL_VARIANTS.items():
        forms = lexical_variant(name, opts, chunks)
        bm25 = lexical.build_bm25(forms)

        def rank(text, bm25=bm25, opts=opts):
            q_lex = lexical.lexical_prep(corrected[by_text[text]], opts["lemmatize"],
                                         opts["remove_stopwords"])
            return lexical.search_bm25(q_lex, bm25, len(chunks))
        for set_name, qs in sets.items():
            rows.append({"branch": "BM25 + typo", "variant": name, "questions": set_name,
                         **_score(rank, qs, qrels, chunks)})
        print(rows[-2], rows[-1], sep="\n", flush=True)

    if not args.no_semantic and idx["embeddings"] is not None:
        info = idx["embeddings"][1]
        model = semantic.load_model(info["model"])
        q_vecs = {t: semantic.embed_query(model, spelling.fix_capitals(t), info["model"])
                  for t in by_text}
        for name, field in SEMANTIC_VARIANTS.items():
            E = semantic_variant(name, field, chunks, idx)

            def rank(text, E=E):
                return semantic.search_semantic(q_vecs[text], E, len(chunks))
            for set_name, qs in sets.items():
                rows.append({"branch": "Semantic (BGE-M3)", "variant": name,
                             "questions": set_name, **_score(rank, qs, qrels, chunks)})
            print(rows[-2], rows[-1], sep="\n", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "ablations.csv", index=False)
    wide = table.pivot_table(index=["branch", "variant"], columns="questions", values="ndcg@10",
                             sort=False)
    (OUT / "ablations.md").write_text(f"# Ablations, nDCG@10\n\n{wide.to_markdown()}\n\n"
                                      f"{table.to_markdown(index=False)}\n", encoding="utf-8")
    print(wide.to_string())


if __name__ == "__main__":
    main()
