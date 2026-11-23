"""Download the Alloprof benchmark and convert it to our corpus and evaluation formats.

Alloprof (lyon-nlp/alloprof, Apache 2.0): French course pages from alloprof.qc.ca and real
student questions, each with its relevant pages. See docs/plan_addendum.md, Part 2.

Writes:
    data/corpus/alloprof/<subject>/<slug>-<id8>.txt   one file per page (text only)
    data/corpus/alloprof/<subject>/.metadata.json     file -> {title, subject, uuid}
    eval/alloprof/queries.json                        [{qid, text, subject, split}]
    eval/alloprof/qrels.csv                           qid,doc_id,grade (grade = 1)

Usage:
    python scripts/fetch_alloprof.py                  # all subjects except "english"
    python scripts/fetch_alloprof.py --keep-english
    python scripts/fetch_alloprof.py --train-sample 300 --seed 42
"""
import argparse
import ast
import csv
import json
import random
import re
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from search import config  # noqa: E402  (sets HF_HUB_CACHE before any HF import)

REPO = "lyon-nlp/alloprof"
EXCLUDED_BY_DEFAULT = {"english"}  # pages that teach English, partly written in English


def slugify(text: str, max_len: int = 50) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:max_len].rstrip("-") or "page"


def load(name: str) -> list[dict]:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, name, repo_type="dataset")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(config.CORPUS_DIR / "alloprof"))
    p.add_argument("--eval-out", default=str(config.EVAL_DIR / "alloprof"))
    p.add_argument("--keep-english", action="store_true")
    p.add_argument("--train-sample", type=int, default=300,
                   help="train questions kept for tuning (alpha, options)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out, eval_out = Path(args.out), Path(args.eval_out)
    excluded = set() if args.keep_english else EXCLUDED_BY_DEFAULT

    print(f"Downloading {REPO} ...", flush=True)
    docs = load("documents.json")
    queries = {"test": load("queries-test.json"), "train": load("queries-train.json")}

    # --- documents -> files ---------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)  # rebuilt from scratch so removed pages do not linger
    uuid_to_doc_id: dict[str, str] = {}
    metadata: dict[str, dict] = {}
    used: set[str] = set()
    kept = Counter()
    for d in docs:
        subject, text = d["topic"], (d.get("text") or "").strip()
        if subject in excluded or not text:
            continue
        name = f"{slugify(d['title'])}-{d['uuid'][:8]}.txt"
        while name in used:  # uuid prefixes are unique in practice; stay safe anyway
            name = name.replace(".txt", "x.txt")
        used.add(name)
        folder = out / subject
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(text + "\n", encoding="utf-8")
        metadata.setdefault(subject, {})[name] = {
            "title": d["title"].strip(), "subject": subject, "uuid": d["uuid"],
            "source": "Alloprof (lyon-nlp/alloprof, Apache 2.0)"}
        uuid_to_doc_id[d["uuid"]] = f"alloprof/{subject}/{name}"
        kept[subject] += 1
    for subject, fields in metadata.items():
        (out / subject / ".metadata.json").write_text(
            json.dumps(fields, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- questions -> queries.json + qrels.csv ----------------------------------------
    rng = random.Random(args.seed)
    train = queries["train"][:]
    rng.shuffle(train)
    rows, qrels, dropped = [], [], Counter()
    for split, items in (("test", queries["test"]), ("train", train)):
        n_split = 0
        for q in items:
            if split == "train" and n_split >= args.train_sample:
                break
            relevant = q["relevant"]
            if isinstance(relevant, str):
                relevant = ast.literal_eval(relevant)  # stored as "['uuid', ...]"
            doc_ids = [uuid_to_doc_id[u] for u in relevant if u in uuid_to_doc_id]
            if not doc_ids:
                dropped[split] += 1  # all relevant pages excluded or missing
                continue
            qid = f"{split}-{q['id']}"
            rows.append({"qid": qid, "text": q["text"].strip(), "subject": q.get("subject", ""),
                         "split": "dev" if split == "train" else "test"})
            qrels.extend((qid, d, 1) for d in doc_ids)
            n_split += 1

    eval_out.mkdir(parents=True, exist_ok=True)
    (eval_out / "queries.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    with (eval_out / "qrels.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "doc_id", "grade"])
        w.writerows(qrels)

    n_test = sum(r["split"] == "test" for r in rows)
    n_dev = sum(r["split"] == "dev" for r in rows)
    print(f"Pages written: {sum(kept.values())} in {out}")
    for subject, n in kept.most_common():
        print(f"  {subject:20} {n}")
    if excluded:
        print(f"Excluded subjects: {', '.join(sorted(excluded))}")
    print(f"Queries: {n_test} test, {n_dev} dev (sampled from train, seed {args.seed}); "
          f"dropped (no relevant page kept): test {dropped['test']}, train {dropped['train']}")
    print(f"Judgments: {len(qrels)} -> {eval_out / 'qrels.csv'}")


if __name__ == "__main__":
    main()
