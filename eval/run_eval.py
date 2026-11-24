"""Phase 5 · Run every system on a query set, score it, write the report tables.

Usage:
    python -m eval.run_eval --dataset alloprof --split dev
    python -m eval.run_eval --dataset alloprof --split test --systems S1 S2 S3 S4 S5
    python -m eval.run_eval --dataset team                   # eval/queries.json + qrels.csv

Writes eval/results/<dataset>_<split>/:
    per_query.csv   one row per (system, query): the three metrics + rank of first relevant
    summary.csv     mean of each metric per system (+ ms / query)
    by_group.csv    nDCG@10 per system and subject (Alloprof) or query type (team)
    summary.md      the same tables in Markdown, ready for the report
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from search import config  # noqa: E402  (sets HF_HUB_CACHE before any HF import)
from search.engine import load_index, search  # noqa: E402
from eval import metrics  # noqa: E402

K = 10

# The systems of the report. S1 has no typo correction: it is the plain baseline.
SYSTEMS: dict[str, dict] = {
    "S1": {"label": "TF-IDF", "mode": "tfidf", "typo_fix": False},
    "S2": {"label": "BM25 + typo", "mode": "bm25"},
    "S2W": {"label": "BM25 + typo + WordNet", "mode": "bm25", "wordnet": True},
    "S3": {"label": "Semantic (BGE-M3)", "mode": "semantic"},
    "S4rrf": {"label": "Hybrid RRF", "mode": "hybrid", "fusion_method": "rrf"},
    "S4": {"label": "Hybrid weighted", "mode": "hybrid"},
    "S5": {"label": "Hybrid + reranker", "mode": "hybrid", "rerank": True},
    "S6": {"label": "Word2vec (mean)", "mode": "word2vec"},
}
DEFAULT_SYSTEMS = ["S1", "S2", "S3", "S4", "S5"]

DATASETS = {
    "alloprof": (config.EVAL_DIR / "alloprof" / "queries.json",
                 config.EVAL_DIR / "alloprof" / "qrels.csv", "subject"),
    "team": (config.EVAL_DIR / "queries.json", config.EVAL_DIR / "qrels.csv", "type"),
}


def load_queries(dataset: str, split: str | None) -> tuple[list[dict], dict[str, dict[str, int]], str]:
    """Queries of one split (None = all) and their judgments {qid: {doc_id: grade}}."""
    q_path, r_path, group_key = DATASETS[dataset]
    queries = json.loads(q_path.read_text(encoding="utf-8"))
    if split:
        queries = [q for q in queries if q.get("split") == split]
    qrels: dict[str, dict[str, int]] = {}
    for row in pd.read_csv(r_path).itertuples(index=False):
        qrels.setdefault(str(row.qid), {})[str(row.doc_id)] = int(row.grade)
    return queries, qrels, group_key


def run_system(name: str, queries: list[dict], qrels: dict[str, dict[str, int]],
               group_key: str, index_dir: Path | None = None, alpha: float | None = None,
               progress: bool = True) -> pd.DataFrame:
    """Score one system on every judged query -> one row per query."""
    kwargs = {k: v for k, v in SYSTEMS[name].items() if k != "label"}
    rows = []
    for i, q in enumerate(queries, start=1):
        judged = qrels.get(q["qid"], {})
        if not metrics.has_relevant(judged):
            continue
        t = time.perf_counter()
        res = search(q["text"], top_k=K, snippets=False, index_dir=index_dir, alpha=alpha,
                     **kwargs)
        ms = (time.perf_counter() - t) * 1000
        ranking = [r["doc_id"] for r in res["results"]]
        first = next((r for r, d in enumerate(ranking, 1) if judged.get(d, 0) > 0), None)
        rows.append({
            "system": name, "qid": q["qid"], "group": q.get(group_key, ""),
            "recall@10": metrics.recall_at_k(ranking, judged, K),
            "mrr@10": metrics.mrr_at_k(ranking, judged, K),
            "ndcg@10": metrics.ndcg_at_k(ranking, judged, K),
            "first_relevant": first, "ms": round(ms, 1), "query_lex": res["query_lex"],
        })
        if progress and i % 100 == 0:
            print(f"  {name}: {i}/{len(queries)}", flush=True)
    return pd.DataFrame(rows)


def summarize(per_query: pd.DataFrame) -> pd.DataFrame:
    s = (per_query.groupby("system", sort=False)
         .agg(n=("qid", "size"), **{m: (m, "mean") for m in ("recall@10", "mrr@10", "ndcg@10")},
              ms=("ms", "median"))
         .round({"recall@10": 3, "mrr@10": 3, "ndcg@10": 3, "ms": 0}))
    s.insert(0, "label", [SYSTEMS[n]["label"] for n in s.index])
    return s


def by_group(per_query: pd.DataFrame) -> pd.DataFrame:
    t = per_query.pivot_table(index="group", columns="system", values="ndcg@10",
                              aggfunc="mean", sort=False).round(3)
    t.insert(0, "n", per_query.groupby("group")["qid"].nunique())
    return t.sort_values("n", ascending=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASETS, default="alloprof")
    p.add_argument("--split", default="dev", help="dev, test, or 'all'")
    p.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS, choices=SYSTEMS)
    p.add_argument("--index", default=None, help="index folder (default config.INDEX_DIR)")
    p.add_argument("--limit", type=int, default=None, help="first N queries only (quick check)")
    p.add_argument("--out", default=None, help="output folder")
    args = p.parse_args()

    split = None if args.split == "all" else args.split
    queries, qrels, group_key = load_queries(args.dataset, split)
    if args.limit:
        queries = queries[:args.limit]
    n_judged = sum(metrics.has_relevant(qrels.get(q["qid"], {})) for q in queries)
    print(f"{args.dataset}/{args.split}: {len(queries)} queries, {n_judged} with a relevant "
          f"document ({len(queries) - n_judged} excluded)")

    index_dir = Path(args.index) if args.index else None
    load_index(index_dir)  # load once, outside the timings
    out = Path(args.out or config.EVAL_DIR / "results" / f"{args.dataset}_{args.split}")
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for name in args.systems:
        t = time.perf_counter()
        frames.append(run_system(name, queries, qrels, group_key, index_dir))
        print(f"{name} done in {time.perf_counter() - t:.0f} s", flush=True)
    per_query = pd.concat(frames, ignore_index=True)

    summary, groups = summarize(per_query), by_group(per_query)
    per_query.to_csv(out / "per_query.csv", index=False)
    summary.to_csv(out / "summary.csv")
    groups.to_csv(out / "by_group.csv")
    (out / "summary.md").write_text(
        f"# {args.dataset} / {args.split}: {n_judged} judged queries\n\n"
        f"{summary.to_markdown()}\n\n## nDCG@10 by {group_key}\n\n{groups.to_markdown()}\n",
        encoding="utf-8")
    print(summary.to_string())
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
