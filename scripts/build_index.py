"""Build the index from a corpus folder and print a summary.

Usage:
    python scripts/build_index.py                                  # data/corpus -> index/
    python scripts/build_index.py --corpus tests/fixtures/corpus   # synthetic test corpus
    python scripts/build_index.py --show 3                         # also print 3 chunks
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from search import config  # noqa: E402  (sets HF_HUB_CACHE before any HF import)
from search.engine import build_index  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(config.CORPUS_DIR))
    p.add_argument("--index", default=str(config.INDEX_DIR))
    p.add_argument("--show", type=int, default=0, help="print the first N chunks")
    args = p.parse_args()

    s = build_index(Path(args.corpus), Path(args.index))
    print(f"{s['n_documents']} documents, {s['n_pages']} pages -> {s['n_chunks']} chunks "
          f"in {s['seconds']} s")
    for path, reason in s["skipped"]:
        print(f"  skipped {path}: {reason}")
    for doc, lines in s["removed_headers_footers"].items():
        print(f"  {doc}: removed {len(lines)} header/footer lines, e.g. {lines[0]!r}")

    if args.show:
        import pandas as pd
        df = pd.read_parquet(Path(args.index) / "chunks.parquet")
        for _, row in df.head(args.show).iterrows():
            print(f"\n--- {row.chunk_id} (page {row.page}, {row.n_words} words)\n{row.text_raw[:400]}")


if __name__ == "__main__":
    main()
